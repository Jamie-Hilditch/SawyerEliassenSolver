module SawyerEliassenSolver

using DocStringExtensions
using Printf
using Reexport

using FFTW: FFTW

include("Utils/Utils.jl")
include("Domains/Domains.jl")
include("Variables/Variables.jl")
include("Forcing/Forcing.jl")
include("Problems/Problems.jl")
include("Timesteppers/Timesteppers.jl")
include("Output/Output.jl")

@reexport using .Domains
@reexport using .Variables
@reexport using .Forcing
@reexport using .Problems
@reexport using .Timesteppers
@reexport using .Output

"""Initialise the SawyerEliassenSolver module."""
function __init__()
    nthreads = Threads.nthreads()
    FFTW.set_num_threads(4nthreads)
    if nthreads > 1
        @info "SawyerEliassenSolver will use $nthreads threads for FFTW"
    end
end

# add a module docstring.
# N.B. to use DocStringExtensions i.e. $(EXPORTS) we need to define the docstring inside the module.

"""
`SawyerEliassenSolver.jl` -- Solve the Sawyer-Eliassen equation using a pseudo-spectral
discretisation and 3rd order accurate implicit timestepping for arbitrary balanced
background flows.

# Exports
$(EXPORTS)
"""
SawyerEliassenSolver

end
