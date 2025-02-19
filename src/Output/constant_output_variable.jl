"""$(TYPEDEF)

A struct to output a constant (time independent) variable.

# Fields
$(FIELDS)

`array` is the array storing the data to be output.
`dimension_labels` is a tuple of dimension labels. When added to an [`OutputWriter`](@ref)
these labels are used to attach dimension scales (coordinates) to the output variable.

!!! note
    Dimension labels are not required to be associated with a dimension scale.
    But if they are then the size of the dimension must match.
"""
struct ConstantOutputVariable{O,N} <: AbstractArray{O,N}
    array::AbstractArray{O,N}
    dimension_labels::NTuple{N,Union{Symbol,Nothing}}
end

Base.size(cov::ConstantOutputVariable) = size(cov.array)
Base.eltype(cov::ConstantOutputVariable) = eltype(cov.array)
Base.ndims(cov::ConstantOutputVariable) = ndims(cov.array)

@propagate_inbounds Base.getindex(cov::ConstantOutputVariable, i::Int) =
    getindex(cov.array, i)
@propagate_inbounds Base.getindex(cov::ConstantOutputVariable, I::Vararg{Int,N}) where {N} =
    getindex(cov.array, I...)
@propagate_inbounds Base.getindex(cov::ConstantOutputVariable, I...) =
    getindex(cov.array, I...)

function create_constant_output_variable!(
    h5::HDF5.File, path::String, cov::ConstantOutputVariable
)
    # Create the dataspace
    dims = size(cov)
    dspace = dataspace(dims, dims)
    chunks = dims
    N = ndims(cov)

    # Create a dataset for the output variable
    var_dset = create_dataset(h5, path, eltype(cov), dspace; chunk=chunks)

    # label dimensions and attach dimension scales
    # Note the low-level API is just a thin wrapper around the C API
    # so we need to use zero-indexed row major indexing i.e. i -> N - i
    for (i, dim) in enumerate(cov.dimension_labels)
        # get dimension name as a string
        dim_name = isnothing(dim) ? "$(path)_$(i)" : String(dim)

        # label the dimension
        label = Vector{UInt8}(dim_name)
        HDF5.API.h5ds_set_label(var_dset.id, N - i, label)

        # if there is no dimension scale associated with this dimension then we are done
        if !haskey(h5, dim_name)
            continue
        end

        # open the dimension scale dataset
        dim_dset = open_dataset(h5, dim_name)

        # check it is actually a scale
        if !HDF5.API.h5ds_is_scale(dim_dset.id)
            @warn "Not attaching dimension $dim_name to $path as it is not a dimension scale"
            continue
        end

        # enforce that the length matches the dimension size
        if length(dim_dset) != dims[i]
            @warn "Not attaching dimension $dim_name as it has length $(length(dim_dset)) but dimension $i on $path has length $(dims[i])"
            continue
        end

        HDF5.API.h5ds_attach_scale(var_dset.id, dim_dset.id, N - i)
    end

    # finally save the data
    indices = ntuple(i -> Colon(), N)
    var_dset[indices...] = cov
    return nothing
end

"""$(TYPEDSIGNATURES)

Write a [`ConstantOutputVariable`](@ref) to the file created by an [`OutputWriter`](@ref).
"""
function write_constant_variable!(
    output_writer::OutputWriter, cov::ConstantOutputVariable, name::String
)
    h5open(output_writer.filepath, "r+") do h5
        create_constant_output_variable!(h5, name, cov)
    end
end

"""$(TYPEDSIGNATURES)

Create a [`ConstantOutputVariable`](@ref) and write it to the file created by an [`OutputWriter`](@ref) in one go.
"""
function write_constant_array!(
    output_writer::OutputWriter,
    array::AbstractArray{T,N},
    name::String,
    dimension_labels::NTuple{N,Union{Symbol,Nothing}}=ntuple(nothing, Val(N)),
) where {T,N}
    cov = ConstantOutputVariable(array, dimension_labels)
    write_constant_variable!(output_writer, cov, name)
    return nothing
end

### Write the background flow to the output writer

"""$(TYPEDSIGNATURES)

Write the background ``f`` as an attribute to the output writer.
"""
function write_f!(output_writer::OutputWriter; name::String="f")
    problem = get_problem(output_writer)
    write_attributes!(output_writer, name => get_f(problem))
    return nothing
end

"""$(TYPEDSIGNATURES)

Write the current background ``V_x`` to the output writer.
"""
function write_Vx!(output_writer::OutputWriter; name::String="Vx")
    problem = get_problem(output_writer)
    write_constant_array!(output_writer, get_Vx(problem), name, (:x, :z))
    return nothing
end

"""$(TYPEDSIGNATURES)

Write the current background ``B_x`` to the output writer.
"""
function write_Bx!(output_writer::OutputWriter; name::String="Bx")
    problem = get_problem(output_writer)
    write_constant_array!(output_writer, get_Bx(problem), name, (:x, :z))
    return nothing
end

"""$(TYPEDSIGNATURES)

Write the current background ``B_z`` to the output writer.
"""
function write_Bz!(output_writer::OutputWriter; name::String="Bz")
    problem = get_problem(output_writer)
    write_constant_array!(output_writer, get_Bz(problem), name, (:x, :z))
    return nothing
end

"""$(TYPEDSIGNATURES)

Write the background flow to the output writer.
"""
function write_background_flow!(
    ow::OutputWriter; f=nothing, Vx=nothing, Bx=nothing, Bz=nothing
)
    isnothing(f) ? write_f!(ow) : write_f!(ow; name=String(f))
    isnothing(Vx) ? write_Vx!(ow) : write_Vx!(ow; name=String(Vx))
    isnothing(Bx) ? write_Bx!(ow) : write_Bx!(ow; name=String(Bx))
    isnothing(Bz) ? write_Bz!(ow) : write_Bz!(ow; name=String(Bz))
    return nothing
end

"""$(TYPEDSIGNATURES)

Integrate the background buoyancy gradients and write to output writer.
"""
function write_background_buoyancy!(ow::OutputWriter; name::String="B")
    problem = get_problem(ow)
    domain = get_domain(problem)
    B = problem.scratch.XZ_tmp

    # get background gradients as XZVariables
    Bx = XZVariable(domain, get_Bx(problem))
    Bz = XZVariable(domain, get_Bz(problem))

    @inbounds integrate_periodic_gradients(Bx, Bz; out=B)

    write_constant_array!(ow, B, name, (:x, :z))
    return nothing
end

"""$(TYPEDSIGNATURES)

Integrate the background velocity gradients and write to output writer.
"""
function write_background_velocity!(ow::OutputWriter; name::String="V")
    problem = get_problem(ow)
    domain = get_domain(problem)
    V = problem.scratch.XZ_tmp

    # get background gradients as XZVariables
    Vx = XZVariable(domain, get_Vx(problem))
    Vz = XZVariable(domain, get_Bx(problem)) ./ get_f(problem)

    @inbounds integrate_periodic_gradients(Vx, Vz; out=V)

    write_constant_array!(ow, V, name, (:x, :z))
    return nothing
end
