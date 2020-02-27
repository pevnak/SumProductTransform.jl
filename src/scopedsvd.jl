using Unitary: Butterfly, TransposedMatVec, TransposedButterfly, DiagonalRectangular
using Unitary: diagmul, _mulax, _mulxa, explicitgrad, _logabsdet

function (m::Unitary.SVDDense{U,D,U,F})(xx::Tuple{A,B,S}) where {U<:Butterfly, D, F, A, B, S<:AbstractScope}
	x, logdet, s = xx
	pre = smul(m.u, smul(m.d, smul(m.v, x, s), s), s) .+ m.b[s.dims]
	g = explicitgrad.(m.σ, pre)
	(m.σ.(pre), logdet .+ sum(log.(g), dims = 1) .+ _logabsdet(m.d, s), s)
end

smul(a::Butterfly, x::TransposedMatVec, s) = _mulax(filter(s, a.θs, a.idxs)..., x, 1)
smul(x::TransposedMatVec, a::Butterfly, s) = _mulxa(x, filter(s, a.θs, a.idxs)..., 1)
smul(a::TransposedButterfly, x::TransposedMatVec, s) = _mulax(filter(s, a.parent.θs, a.parent.idxs)..., x, -1)
smul(x::TransposedMatVec, a::TransposedButterfly, s) = _mulxa(x, filter(s, a.parent.θs, a.parent.idxs)..., -1)

smul(a::DiagonalRectangular, x::TransposedMatVec, s) = diagmul(a.d[s.dims], size(x,1), size(x,1), x)
Unitary._logabsdet(a::DiagonalRectangular, s::AbstractScope) = sum(log.(abs.(a.d[s.dims]) .+ eps(eltype(a.d))))