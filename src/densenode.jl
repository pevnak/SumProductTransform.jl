struct DenseNode{M,MI,P} <: Distributions.ContinuousMultivariateDistribution
	m::M
	mi::MI
	p::P 
end

Base.length(m::DenseNode) = length(m.p)
Flux.@treelike(DenseNode)
DenseNode(m, p) = DenseNode(m, inv(m), p)


function Distributions.logpdf(m::DenseNode, x::AbstractMatrix{T}) where {T}
	x, l = m.m((x,zero(T)))
	logpdf(m.p, x) .+ l[:]
end

function Distributions.logpdf(m::M, x::AbstractMatrix) where {M<: MvNormal{T,Distributions.PDMats.ScalMat{T},Distributions.ZeroVector{T}}} where {T}
	log_normal(x, m.μ)[:]
end

function pathlogpdf(m::DenseNode, x::AbstractMatrix{T}, path) where {T}
	x, l = m.m((x,zero(T)))
	pathlogpdf(m.p, x, path) .+ l[:]
end

pathcount(m::DenseNode) = pathcount(m.p)

samplepath(m::DenseNode) = samplepath(m.p)

function mappath(m::DenseNode, x::AbstractMatrix{T}) where {T}
	x, l = m.m((x,zero(T)))
	lkl, path = mappath(m.p, x)
	return(lkl .+ l[:], path)
end


Base.rand(m::DenseNode) = m.mi(rand(m.p))

Base.show(io::IO, m::DenseNode{M,MI,P}) where {M,MI,P} = dsprint(io, m)
function dsprint(io::IO, n::DenseNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → ", color=c)
    dsprint(io, n.p, pad=[pad; (c, "     ")])
end
function dsprint(io::IO, n::DenseNode{M,MI,P}; pad=[]) where {M,MI,P<:MvNormal}
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → MvNormal\n", color=c)
end

