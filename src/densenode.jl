struct DenseNode{M,P} <: Distributions.ContinuousMultivariateDistribution
	m::M
	p::P 
end

Base.length(m::DenseNode) = length(m.p)
Flux.@treelike(DenseNode)


Distributions.logpdf(m::DenseNode, x::AbstractVector) = logpdf(m, reshape(x, :, 1))[1]
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
_updatelatent!(m::DenseNode, path) = _updatelatent!(m.p, path)
zerolatent!(m::DenseNode) = zerolatent!(m.p)
normalizelatent!(m::DenseNode) = normalizelatent!(m.p)

function mappath(m::DenseNode, x::AbstractMatrix{T}) where {T}
	x, l = m.m((x,zero(T)))
	lkl, path = mappath(m.p, x)
	return(lkl .+ l[:], path)
end


Base.rand(m::DenseNode) = inv(m.m)(rand(m.p))

Base.show(io::IO, m::DenseNode{M,P}) where {M,P} = dsprint(io, m)
function dsprint(io::IO, n::DenseNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → ", color=c)
    dsprint(io, n.p, pad=[pad; (c, "     ")])
end
function dsprint(io::IO, n::DenseNode{M,P}; pad=[]) where {M,P<:MvNormal}
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → MvNormal\n", color=c)
end

