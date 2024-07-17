const CG_TOL_DEFAULT::Float64 = 10^-10 #10^-5

struct ConjugateGradientSolver{T}
    domain::Domain{T}
    p::FSVariable{T}
    q::FSVariable{T}
    r::FSVariable{T}
    z::FSVariable{T}
    max_iterations::Int
    tol::T

    function ConjugateGradientSolver(
        domain::Domain{T}, max_iterations::Int, cg_tol::T
    ) where {T}
        cg_tol > 0 || throw(DomainError(cg_tol, "tolerance must be positive"))
        return new{T}(
            domain,
            FSVariable(domain),
            FSVariable(domain),
            FSVariable(domain),
            FSVariable(domain),
            max_iterations,
            cg_tol,
        )
    end
end

function ConjugateGradientSolver(domain::Domain{T}, ::Nothing, cg_tol::T) where {T}
    # without roundoff error the conjugate gradient method converges in at most n iterations
    # where n is the dimension of the problem
    max_iterations_default = prod(size(domain.spectral))
    return ConjugateGradientSolver(domain, max_iterations_default, cg_tol)
end

function ConjugateGradientSolver(
    domain::Domain{T}, max_iterations::Int, ::Nothing
) where {T}
    return ConjugateGradientSolver(domain, max_iterations, convert(T, CG_TOL_DEFAULT))
end

function ConjugateGradientSolver(domain::Domain{T}, ::Nothing, ::Nothing) where {T}
    return ConjugateGradientSolver(
        domain, nothing, convert(T, CG_TOL_DEFAULT)
    )
end

Domains.get_domain(cgs::ConjugateGradientSolver) = cgs.domain

@inline function solve_implicit_equation!(
    cgs::ConjugateGradientSolver{T},
    𝓛ᴵ!::ImplicitSawyerEliassenOperator!,
    x::FSVariable{T},
    b::FSVariable{T},
    𝓟::AbstractPreconditioner{T},
) where {T}
    # some setup before we begin
    # extract variables from cg and A
    (; domain, p, q, r, z, max_iterations, tol) = cgs
    aᵢᵢ, h = 𝓛ᴵ!.aᵢᵢ, 𝓛ᴵ!.h

    @boundscheck consistent_domains(domain, 𝓛ᴵ!, x, b, 𝓟) || throw(
        ArgumentError("`cgs`, `𝓛ᴵ!`, `x`, `b` and `𝓟` must have the same domain.")
    )

    # termination condition
    condition = tol * real(dot(b, b))
    @debug "Stopping condition = $(condition)"

    # compute the explicit residual every √max_iterations iterations
    explicit_residual_cadence = isqrt(max_iterations)

    # ################################ #
    # The conjugate gradient algorithm #
    # ################################ #

    # Notation:
    #   A is the linear operator
    #   M is the preconditioner
    #   xₖ is the current solution
    #   pₖ is the search direction
    #   rₖ = b - Axₖ is the residual
    #   qₖ = Apₖ
    #   zₖ = M⁻¹rₖ is the preconditioned residual
    #   δₖ = rₖᵀzₖ = rₖᵀM⁻¹rₖ is the M⁻¹ error of the residual
    #   αₖ = δₖ / pₖᵀApₖ is the step size

    # initialise the variables
    @inbounds 𝓛ᴵ!(q, x)
    @inbounds @. r = b - q # r₀ = b - Ax₀
    @inbounds apply_preconditioner!(𝓟, z, r, aᵢᵢ, h) # Mz₀ = r₀
    δ = r ⋅ z # δ = r₀ᵀz₀
    @inbounds @. p = z # p₀ = z₀

    # the main event
    for k in 1:max_iterations
        @inbounds 𝓛ᴵ!(q, p) # qₖ = Apₖ
        α = δ / (p ⋅ q) # αₖ = rₖᵀzₖ / pₖᵀApₖ = δₖ / pₖᵀqₖ
        @inbounds @. x += α * p # xₖ₊₁ = xₖ + αₖpₖ
        # compute residual -- occasionally explicitly to avoid round-off error
        if k % explicit_residual_cadence != 0
            @inbounds @. r -= α * q # rₖ₊₁ = rₖ - αₖApₖ = rₖ - αₖqₖ
        else
            @inbounds 𝓛ᴵ!(q, x)
            @inbounds @. r = b - q # rₖ₊₁ = b - Axₖ₊₁
        end
        # terminate?
        if real(r ⋅ r) < condition # rₖ₊₁ᵀrₖ₊₁ < tol * bᵀb
            @debug "Conjugate gradient algorithm completed in $(k) iterations"
            return nothing
        end
        # get δₖ₊₁ = rₖ₊₁ᵀzₖ₊₁
        @inbounds apply_preconditioner!(𝓟, z, r, aᵢᵢ, h) # Mzₖ₊₁ = rₖ₊₁
        δ⁺ = r ⋅ z # δₖ₊₁ = rₖ₊₁ᵀzₖ₊₁
        # update the search direction
        β = δ⁺ / δ # βₖ = rₖ₊₁ᵀzₖ₊₁ / rₖᵀzₖ = δₖ₊₁ / δₖ
        @inbounds @. p = z + β * p # pₖ₊₁ = zₖ₊₁ + βₖpₖ
        # update δₖ₊₁
        δ = δ⁺
    end
    @warn "Conjugate gradient solver did not terminate after $(max_iterations) iterations. Stopping condition = $(condition)."
    return nothing
end
