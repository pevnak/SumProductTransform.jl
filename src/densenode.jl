struct DenseNode{M,MI,P} <: Distributions.ContinuousMultivariateDistribution
	m::M
	mi::MI
	p::P 
end

Base.show(io::IO, m::DenseNode{M,MI,P}) where {M,MI,P} = dsprint(io, m)
function dsprint(io::IO, n::DenseNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) →\n", color=c, pad=pad)
    dsprint(io, n.p, pad=[pad; (c, "     ")])
end
function dsprint(io::IO, n::DenseNode{M,MI,P}; pad=[]) where {M,MI,P<:MvNormal}
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → MvNormal\n", color=c, pad=pad)
end

Flux.@treelike(DenseNode)

DenseNode(m, p) = DenseNode(m, inv(m), p)

function Distributions.logpdf(m::DenseNode, x::AbstractMatrix)
	x, l = m.m((x,0))
	logpdf(m.p, x) .+ l[:]
end

function Distributions.logpdf(m::M, x::AbstractMatrix) where {M<: MvNormal{T,Distributions.PDMats.ScalMat{T},Distributions.ZeroVector{T}}} where {T}
	log_normal(x, m.μ)[:]
end
