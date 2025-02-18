const DEFAULT_C::Float64 = 17 / 14

struct DIRKNCoefficients{T}
    a₁₁::T
    a₂₁::T
    a₂₂::T
    b₁::T
    b₂::T
    b₁ᵗ::T
    b₂ᵗ::T
    c₁::T
    c₂::T

    function DIRKNCoefficients(c::T) where {T}
        denom = 4 * (3 * c^2 - 3 * c + 1)

        a₁₁ = c^2 / 2
        a₂₁ = -2 * (9 * c^4 - 9 * c^3 + 3 * c - 1) / (9 * (2 * c - 1)^2)
        a₂₂ = c^2 / 2
        b₁ = (1 - c) / denom
        b₂ = (3 * c - 1) * (2 * c - 1) / denom
        b₁ᵗ = 1 / denom
        b₂ᵗ = 3 * (2 * c - 1)^2 / denom
        c₁ = c
        c₂ = (3 * c - 2) / (3 * (2 * c - 1))
        return new{T}(a₁₁, a₂₁, a₂₂, b₁, b₂, b₁ᵗ, b₂ᵗ, c₁, c₂)
    end
end

DIRKNCoefficients(::Type{T}) where {T} = DIRKNCoefficients(convert(T, DEFAULT_C))

"""Auxiliary variables for computing ζⁿ⁺¹ and ζₜⁿ⁺¹"""
struct AuxiliaryVariables{T}
    ζⁿ⁺ᶜ¹::FSVariable{T}
    ζⁿ⁺ᶜ²::FSVariable{T}
    tmp::FSVariable{T}
    rhs::FSVariable{T}
end

function AuxiliaryVariables(domain::Domain{T}) where {T}
    ζⁿ⁺ᶜ¹ = FSVariable(domain)
    ζⁿ⁺ᶜ² = FSVariable(domain)
    tmp = FSVariable(domain)
    rhs = FSVariable(domain)
    return AuxiliaryVariables{T}(ζⁿ⁺ᶜ¹, ζⁿ⁺ᶜ², tmp, rhs)
end

"""$(TYPEDEF)
Object that stores all the variables and operators required to advance a problem one timestep.
"""
struct Timestepper{T,F,G,H,P}
    problem::Problem{T,F,G,H}
    h::T
    𝓒::DIRKNCoefficients{T}
    auxiliary_variables::AuxiliaryVariables{T}
    cgs::ConjugateGradientSolver{T}
    𝓟::P

    function Timestepper(
        problem::Problem{T,F,G,H},
        h::T,
        𝓒::DIRKNCoefficients{T},
        cgs::ConjugateGradientSolver{T},
        𝓟::AbstractPreconditioner{T},
    ) where {T,F,G,H}
        consistent_domains(problem, cgs, 𝓟) ||
            throw(ArgumentError("`problem`, `cgs` and `𝓟` must have the same domain."))

        domain = get_domain(problem)
        auxiliary_variables = AuxiliaryVariables(domain)

        return new{T,F,G,H,typeof(𝓟)}(problem, h, 𝓒, auxiliary_variables, cgs, 𝓟)
    end
end

"""
```
Timestepper(problem::Problem{T}, h::T, [𝓟::AbstractPreconditioner]; c=nothing, cg_max_iterations=nothing, cg_tol=nothing)
```

Constructor for `Timestepper`. The timestep `h` is required and a preconditioner may optionally
be passed.

# Keyword arguments

* `c`: use a non-default value for the free parameter in the timestepping stepping scheme. See [Sharp_Fine_Burrage_1990](@citet) for valid ranges of values.
* `cg_max_iterations`: maximum number of iterations for the conjugate gradient solver.
* `cg_tol`: tolerance for the conjugate gradient solver.
"""
function Timestepper(
    problem::Problem{T},
    h::T,
    𝓟=nothing;
    c=nothing,
    cg_max_iterations=nothing,
    cg_tol=nothing,
) where {T}
    𝓒 = isnothing(c) ? DIRKNCoefficients(T) : DIRKNCoefficients(c)
    aᵢᵢh² = 𝓒.a₁₁ * h^2
    domain = get_domain(problem)
    cgs = ConjugateGradientSolver(domain, aᵢᵢh², cg_max_iterations, cg_tol)
    if isnothing(𝓟)
        𝓟 = IdentityPreconditioner(domain)
    end
    return Timestepper(problem, h, 𝓒, cgs, 𝓟)
end

function Base.show(io::IO, ::MIME"text/plain", ts::Timestepper)
    return print(
        io,
        "Timestepper:\n",
        "  ├───────── problem: $(summary(ts.problem))\n",
        "  ├──────── timestep: h = $(sfmt(ts.h))\n",
        "  ├─────────────── 𝓒: $(summary(ts.𝓒))\n",
        "  ├───────────── cgs: $(summary(ts.cgs))\n",
        "  └─────────────── 𝓟: $(summary(ts.𝓟))\n",
    )
end

function Base.summary(io::IO, ts::Timestepper)
    return print(io, "Timestepper with timestep $(sfmt(ts.h))")
end

"""$(TYPEDSIGNATURES)"""
Problems.get_problem(ts::Timestepper) = ts.problem
