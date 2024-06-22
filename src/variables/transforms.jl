## Unsafe horizontal transforms for internal use. ##

"""Unsafe horizontal transformation from physical to Fourier space."""
@inline function Tᴴ!(out::FVariable{T}, in::XVariable{T}) where {T}
    fourier = in.domain.transforms.fourier
    mul!(out.data, fourier, in.data)
    # zero out the high wavenumbers for dealiasing
    CNX = out.domain.spectral.CNX
    @inbounds out[(CNX + 1):end, :] .= 0
end

"""Unsafe horizontal transformation from Fourier to physical space."""
@inline function Tᴴ!(out::XVariable{T}, in::FVariable{T}) where {T}
    fourier = in.domain.transforms.fourier
    return ldiv!(out.data, fourier, in.data)
end

## Unsafe sine transform for internal use. ##

"""Unsafe sine transformation from physical to spectral space."""
@inline function Tˢ!(out::SVariable{T}, in::ZVariable{T}) where {T}
    sine = in.domain.transforms.sine
    mul!(out.data, sine, in.data)
    # zero out the high wavenumbers for dealiasing
    CNZ = out.domain.spectral.CNZ
    @inbounds out[:, (CNZ + 1):end] .= 0
end

"""Unsafe sine transformation from spectral to physical space."""
@inline function Tˢ!(out::ZVariable{T}, in::SVariable{T}) where {T}
    sine = in.domain.transforms.sine
    return ldiv!(out.data, sine, in.data)
end

## Unsafe cosine transform for internal use. ##

"""Unsafe cosine transformation from physical to spectral space."""
@inline function Tᶜ!(out::CVariable{T}, in::ZVariable{T}) where {T}
    cosine = in.domain.transforms.cosine
    mul!(out.data, cosine, in.data)
    # zero out the high wavenumbers for dealiasing
    CNZ = out.domain.spectral.CNZ
    @inbounds out[:, (CNZ + 1):end] .= 0
end

"""Unsafe cosine transformation from spectral to physical space."""
@inline function Tˢ!(out::ZVariable{T}, in::CVariable{T}) where {T}
    cosine = in.domain.transforms.cosine
    return ldiv!(out.data, cosine, in.data)
end

## all valid combinations of inplace transformations ##
# horizontal
@inline _transform!(out::FZVariable{T}, in::XZVariable{T}) where {T} = Tᴴ!(out, in)
@inline _transform!(out::FSVariable{T}, in::XSVariable{T}) where {T} = Tᴴ!(out, in)
@inline _transform!(out::FCVariable{T}, in::XCVariable{T}) where {T} = Tᴴ!(out, in)
@inline _transform!(out::XZVariable{T}, in::FZVariable{T}) where {T} = Tᴴ!(out, in)
@inline _transform!(out::XSVariable{T}, in::FSVariable{T}) where {T} = Tᴴ!(out, in)
@inline _transform!(out::XCVariable{T}, in::FCVariable{T}) where {T} = Tᴴ!(out, in)
# sine
@inline _transform!(out::XSVariable{T}, in::XZVariable{T}) where {T} = Tˢ!(out, in)
@inline _transform!(out::FSVariable{T}, in::FZVariable{T}) where {T} = Tˢ!(out, in)
@inline _transform!(out::XZVariable{T}, in::XSVariable{T}) where {T} = Tˢ!(out, in)
@inline _transform!(out::FZVariable{T}, in::FSVariable{T}) where {T} = Tˢ!(out, in)
# cosine
@inline _transform!(out::XCVariable{T}, in::XZVariable{T}) where {T} = Tᶜ!(out, in)
@inline _transform!(out::FCVariable{T}, in::FZVariable{T}) where {T} = Tᶜ!(out, in)
@inline _transform!(out::XZVariable{T}, in::XCVariable{T}) where {T} = Tᶜ!(out, in)
@inline _transform!(out::FZVariable{T}, in::FCVariable{T}) where {T} = Tᶜ!(out, in)

######################
## Public functions ##
######################

"""Safe transforms with domain validation for public use."""
function transform!(out::AbstractVariable, in::AbstractVariable)
    in.domain === out.domain || error("in and out must have the same domain")
    return _transform!(out, in)
end

# horizontal transformations creating the output variable
horizontal_transform(in::XZVariable) = Tᴴ!(FZVariable(in.domain), in)
horizontal_transform(in::XSVariable) = Tᴴ!(FSVariable(in.domain), in)
horizontal_transform(in::XCVariable) = Tᴴ!(FCVariable(in.domain), in)
horizontal_transform(in::FZVariable) = Tᴴ!(XZVariable(in.domain), in)
horizontal_transform(in::FSVariable) = Tᴴ!(XSVariable(in.domain), in)
horizontal_transform(in::FCVariable) = Tᴴ!(XCVariable(in.domain), in)
horizontal_transform!(out, in) = transform!(out, in)

# sine transforms creating the output variable
sine_transform(in::XZVariable) = Tˢ!(XSVariable(in.domain), in)
sine_transform(in::XSVariable) = Tˢ!(XZVariable(in.domain), in)
sine_transform(in::FZVariable) = Tˢ!(FSVariable(in.domain), in)
sine_transform(in::FSVariable) = Tˢ!(FZVariable(in.domain), in)
sine_transform!(out, in) = transform!(out, in)

# cosine transforms creating the output variable
cosine_transform(in::XZVariable) = Tᶜ!(XCVariable(in.domain), in)
cosine_transform(in::XCVariable) = Tᶜ!(XZVariable(in.domain), in)
cosine_transform(in::FZVariable) = Tᶜ!(FCVariable(in.domain), in)
cosine_transform(in::FCVariable) = Tᶜ!(FZVariable(in.domain), in)
cosine_transform!(out, in) = transform!(out, in)
