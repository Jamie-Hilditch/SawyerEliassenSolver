"""Size of spectral domain and wavenumbers
"""
struct Spectral{T<:SingleOrDouble}
    SX::Int
    NZ::Int
    CNX::Int
    CNZ::Int
    Δkx::T
    Δkz::T
end

function Spectral(grid::Grid{T}, CNX, CNZ) where {T}
    SX = grid.NX ÷ 2 + 1
    LX = grid.x_bounds[2] - grid.x_bounds[1]
    LZ = grid.z_bounds[2] - grid.z_bounds[1]
    Δkx = 2 * π / LX
    Δkz = π / LZ
    return Spectral{T}(SX, grid.NZ, CNX, CNZ, Δkx, Δkz)
end

Base.eltype(::Spectral{T}) where {T} = T
Base.size(spectral::Spectral) = (spectral.SX, spectral.NZ)

function Base.summary(io::IO, spectral::Spectral)
    return print(
        io,
        "Spectral domain of size $(size(spectral)) and spectral resolution $((spectral.CNX,spectral.CNZ))",
    )
end

@inline xwavenumbers(spectral::Spectral) = spectral.Δkx * UnitRange(0, spectral.CNX - 1)
@inline zwavenumbers(spectral::Spectral) = spectral.Δkz * UnitRange(1, spectral.CNZ)
@inline function wavenumbers(spectral::Spectral)
    kx = xwavenumbers(spectral)
    kz = zwavenumbers(spectral)
    return reshape(kx, :, 1), reshape(kz, 1, :)
end
@inline xwavenumbers_full(spectral::Spectral) = spectral.Δkx * UnitRange(0, spectral.SX - 1)
@inline zwavenumbers_full(spectral::Spectral) = spectral.Δkz * UnitRange(1, spectral.NZ)
@inline function wavenumbers_full(spectral::Spectral)
    kx = xwavenumbers_full(spectral)
    kz = zwavenumbers_full(spectral)
    return reshape(kx, :, 1), reshape(kz, 1, :)
end

@inline resolved_fourier_indices(spectral::Spectral) = 1:(spectral.CNX)
@inline resolved_sine_indices(spectral::Spectral) = 1:(spectral.CNZ)
@inline resolved_cosine_indices(spectral::Spectral) = 2:(spectral.CNZ + 1)
