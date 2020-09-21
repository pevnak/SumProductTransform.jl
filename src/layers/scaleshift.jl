struct ScaleShift{D, B, S}
	d::D
	b::B
	σ::S
end

Base.show(io::IO, m::ScaleShift) = print(io, "ScaleShift{$(size(m.d)), $(m.σ)}")

Flux.@functor ScaleShift

"""
	ScaleShift(n, σ)

	scales the input variables and shift them. Optionally, they are preprocessed by non-linearity.
	
	`σ` --- an invertible and transfer function, cuurently implemented `selu` and `identity`
"""
function ScaleShift(n::Int, σ = identity)
	ScaleShift(DiagonalRectangular(rand(Float32, n), n, n),
		randn(Float32,n),
		σ)
end

function _transform(m::ScaleShift, x)
    z = m.d * x .+ m.b
    (transformed = m.σ.(z), z = z)
end

(m::ScaleShift)(x::AbstractMatVec) = _transform(m, x)[1]

function forward(m::ScaleShift, x::AbstractVecOrMat)
    transformed, z = _transform(m, x)
    g = _explicitgrad.(m.σ, z)
    logabsdetjac = sum(log.(g), dims = 1) .+ _logabsdet(m.d)
    return (rv = transformed, logabsdetjac = logabsdetjac)
end

logabsdetjac(m::ScaleShift, x) = forward(m, x).logabsdetjac



struct InvertedScaleShift{D, B, S}
	d::D
	b::B
	σ::S
end
Flux.@functor InvertedScaleShift

Base.inv(m::ScaleShift) = InvertedScaleShift(inv(m.d), m.b, inv(m.σ))
Base.inv(m::InvertedScaleShift) = ScaleShift(inv(m.d), m.b, inv(m.σ))

function _transform(m::InvertedScaleShift, x)
    m.d * (m.σ.(x) .- m.b)
end

(m::InvertedScaleShift)(x::AbstractMatVec) = _transform(m, x)

function forward(m::InvertedScaleShift, x::AbstractVecOrMat)
	transformed = _transform(m, x)
	g = _explicitgrad.(m.σ, x)
	logabsdetjac = sum(log.(g), dims = 1) .+ _logabsdet(m.d)
    return (rv = transformed, logabsdetjac = logabsdetjac)
end

logabsdetjac(m::InvertedScaleShift, x) = forward(m, x).logabsdetjac
