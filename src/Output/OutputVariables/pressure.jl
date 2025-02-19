raw"""Compute the (kinematic) pressure gradient from the u and w momentum equations.

``\frac{\partial u}{\partial t} - fv = -\frac{\partial p}{\partial x}``
``\frac{\partial w}{\partial t} - b  = -\frac{\partial p}{\partial z}``

Returns ∂p/∂x stored in the scratch.XZ_tmp variable and ∂p/∂z stored in the
scratch.XZ_tmp2 variable.
"""
function _compute_pressure_gradients!(problem::Problem)
    scratch = get_scratch(problem)
    ∂p∂x = scratch.XZ_tmp
    ∂p∂z = scratch.XZ_tmp2

    # first compute Ψₜ from ζₜ
    Ψₜ = scratch.FS_tmp
    ∇⁻²!(Ψₜ, problem.state.ζₜ)

    # now compute -uₜ = ∂Ψₜ/∂z in FC space
    @inbounds ∂z!(scratch.FC_tmp, Ψₜ)

    # transform to XZ
    Tᴴ!(scratch.XC_tmp, scratch.FC_tmp)
    Tᶜ!(∂p∂x, scratch.XC_tmp)

    # add on fv
    @inbounds @. ∂p∂x += problem.state.v * get_f(problem)

    # now compute wₜ = ∂Ψₜ/∂x in FS space
    @inbounds ∂x!(scratch.FS_tmp, Ψₜ)

    # transform to XZ
    Tᴴ!(scratch.XS_tmp, scratch.FS_tmp)
    Tˢ!(∂p∂z, scratch.XS_tmp)

    # negate to get ∂p∂z
    ∂p∂z .*= -1

    # add on b
    @inbounds @. ∂p∂z += problem.state.b

    return ∂p∂x, ∂p∂z
end

"""$(TYPEDSIGNATURES)"""
function _compute_p!(problem::Problem{T}, p::XZVariable{T}) where {T}
    @boundscheck consistent_domains(problem, p)
    scratch = get_scratch(problem)

    # compute gradients
    # these are stored in scratch.XZ_tmp and scratch.XZ_tmp2
    ∂p∂x, ∂p∂z = _compute_pressure_gradients!(problem)
    # integrate gradients
    integrate_periodic_gradients(∂p∂x, ∂p∂z; out=p, FZ_working_space=scratch.FZ_tmp)
    return p
end

@inline _compute_p!(problem, p) = @inbounds _compute_p!(problem, p)

"""$(TYPEDSIGNATURES)"""
p(problem::Problem) = OutputVariable(problem, _compute_p!, (:x, :z), problem.scratch.XZ_tmp)
