struct LUDense{M, B, S}
	m::M
	b::B
	σ::S
end

Base.show(io::IO, a::LUDense) = print(io, "LUDense{$(size(a.m)), $(a.σ)}")

Flux.@functor LUDense

"""
	LUDense(n, σ)

	Dense layer with square weight matrix of dimension `n` parametrized in 
	LU or LDU decomposition.
	
	`σ` --- an invertible and transfer function, curently implemented `selu` and `identity`
"""
function LUDense(n::Int, σ = identity, decom = :ldu)
	n == 1 && return(ScaleShift(1, σ))
	if decom == :lu
		return(_ludense(n, σ))
	elseif decom == :ldu
		return(_ldudense(n, σ))
	else
		@error "unknown type of decompostion $decom"
	end
end


using LinearAlgebra

_ludense(n::Int, σ) = LUDense(lowup(Float32, n), 0.01f0.*randn(Float32,n), σ)
_ldudense(n::Int, σ) = LUDense(lowdup(Float32, n), 0.01f0.*randn(Float32,n), σ)

function _transform(m::LUDense, x)
    z = m.m * x .+ m.b
    (transformed = m.σ.(z), z = z)
end

(m::LUDense)(x::Unitary.AbstractMatVec) = _transform(m, x).transformed


function forward(m::LUDense, x::AbstractVecOrMat)
    transformed, z = _transform(m, x)

    g = _explicitgrad.(m.σ, z)
    logabsdetjac = sum(log.(g), dims = 1) .+ _logabsdet(m.m)

    return (rv = transformed, logabsdetjac = logabsdetjac)
end

logabsdetjac(m::LUDense, x) = forward(m, x).logabsdetjac

struct InvertedLUDense{M, B, S}
	m::M
	b::B
	σ::S
end
Flux.@functor InvertedLUDense

Base.inv(a::LUDense) = InvertedLUDense(inv(a.m), a.b, inv(a.σ))
Base.inv(a::InvertedLUDense) = LUDense(inv(a.m), a.b, inv(a.σ))

function _transform(m::InvertedLUDense, x)
    (m.m * (m.σ.(x) .- m.b))
end

(m::InvertedLUDense)(x::Unitary.AbstractMatVec) = _transform(m, x)

function forward(m::InvertedLUDense, x::AbstractVecOrMat)
    transformed = _transform(m, x)

    g = _explicitgrad.(m.σ, x)
    logabsdetjac = sum(log.(g), dims = 1) .+ _logabsdet(m.m)

    return (rv = transformed, logabsdetjac = logabsdetjac)
end

logabsdetjac(m::InvertedLUDense, x) = forward(m, x).logabsdetjac
