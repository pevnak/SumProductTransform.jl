using Unitary: Butterfly, TransposedMatVec, TransposedButterfly, DiagonalRectangular, ScaleShift
using Unitary: diagmul, _mulax, _mulxa, explicitgrad, _logabsdet

function (m::Unitary.SVDDense{U,D,U,F})(xx::Tuple{A,B,S}) where {U<:Butterfly, D, F, A, B, S<:AbstractScope}
	x, logdet, s = xx
	pre = smul(m.u, smul(m.d, smul(m.v, x, s), s), s) .+ m.b[s.dims]
	g = explicitgrad.(m.σ, pre)
	(m.σ.(pre), logdet .+ sum(log.(g), dims = 1) .+ _logabsdet(m.d, s), s)
end

function (m::Unitary.SVDDense{U,D,U,F})(xx::Tuple{A,B,S}) where {U<:Butterfly, D, F, A, B, S<:NoScope}
	x, logdet, s = xx
	return(m((x, logdet))...,s)
end

function (m::Unitary.ScaleShift{U,D,F})(xx::Tuple{A,B,S}) where {U<:DiagonalRectangular, D, F, A, B, S<:NoScope}
	x, logdet, s = xx
	return(m((x, logdet))...,s)
end

smul(a::Butterfly, x::TransposedMatVec, s) = _mulax(filter(s, a.θs, a.idxs)..., x, 1)
smul(x::TransposedMatVec, a::Butterfly, s) = _mulxa(x, filter(s, a.θs, a.idxs)..., 1)
smul(a::TransposedButterfly, x::TransposedMatVec, s) = _mulax(filter(s, a.parent.θs, a.parent.idxs)..., x, -1)
smul(x::TransposedMatVec, a::TransposedButterfly, s) = _mulxa(x, filter(s, a.parent.θs, a.parent.idxs)..., -1)

smul(a::DiagonalRectangular, x::TransposedMatVec, s) = diagmul(a.d[s.dims], size(x,1), size(x,1), x)
Unitary._logabsdet(a::DiagonalRectangular, s::AbstractScope) = sum(log.(abs.(a.d[s.dims]) .+ eps(eltype(a.d))))


Base.getindex(a::Butterfly, s::Scope) = 
	Butterfly(filter(s,a.θs, a.idxs)..., length(s.dims))
Base.getindex(a::DiagonalRectangular, s::Scope) = 
	DiagonalRectangular(a.d[s.dims], length(s.dims), length(s.dims))
Base.getindex(a::SVDDense, s::Scope) = 
	SVDDense(a.u[s], a.d[s], a.v[s], a.b[s.dims], a.σ)


function (m::Unitary.LUDense)(xx::Tuple{A,B,S}) where {A, B, S<:NoScope}
	x, logdet, s = xx
	return(m((x, logdet))...,s)
end
