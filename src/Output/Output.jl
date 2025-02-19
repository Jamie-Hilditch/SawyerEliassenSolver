"""
"""
module Output

using Base: @propagate_inbounds

using DocStringExtensions
using HDF5
using Statistics

using SawyerEliassenSolver.Domains
using SawyerEliassenSolver.Variables
using SawyerEliassenSolver.Problems

using SawyerEliassenSolver.Variables: Tᴴ!, Tˢ!, Tᶜ!

export OutputVariable,
    compute!,
    OutputWriter,
    add_output_variables!,
    write!,
    write_attributes!,
    OutputVariables,
    ConstantOutputVariable,
    write_constant_array!,
    write_f!,
    write_Vx!,
    write_Bx!,
    write_Bz!,
    write_background_flow!,
    integrate_periodic_gradients,
    write_background_buoyancy!,
    write_background_velocity!

include("output_variable.jl")
include("output_writer.jl")
include("integrate_gradients.jl")
include("constant_output_variable.jl")
include("OutputVariables/OutputVariables.jl")

end  # module Output
