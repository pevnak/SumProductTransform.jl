using Unitary: lowup, TransposedMatVec
using Unitary: mulaxlu, mulxalu, explicitgrad, _logabsdet

function (a::Unitary.LUDense{M, B, F})(xx::Tuple{X, Y, S}) where {M, B, F, X, Y, S<:AbstractScope}
	x, logdet, s = xx
	pre = smulax(a.m, x, s) .+ filter(s, a.b)
	g = explicitgrad.(a.σ, pre)
	(a.σ.(pre), logdet .+ sum(log.(g), dims = 1) .+ _logabsdet(a.m, s), s)
end

Base.filter(s::FullScope, m::Matrix) = m
Base.filter(s::NoScope, m::Matrix) = m
Base.filter(s::Scope, m::Matrix) = m.*(s.active*s.active')+I(s.n).*.!(s.active*s.active')
Base.filter(s::FullScope, v::Vector) = v
Base.filter(s::NoScope, v::Vector) = v
Base.filter(s::Scope, v::Vector) = v.*s.active

smulax(a::lowup, x::TransposedMatVec, s) = mulaxlu(filter(s, a.m), x, a.invs)
smulxa(a::lowup, x::TransposedMatVec, s) = mulxalu(filter(s, a.m), x, a.invs)

function Unitary._logabsdet(a::lowup{T}, s::Scope) where {T<:Number}
	out = zero(T)
	for i = 1:s.dims
		out += log(abs(a.m[i, i] + eps(T)))
	end
	out
end
function Unitary._logabsdet(a::lowup{T}, s::AbstractScope) where {T<:Number}
	out = zero(T)
	for i = 1:a.n
		out += log(abs(a.m[i, i] + eps(T)))
	end
	out
end
