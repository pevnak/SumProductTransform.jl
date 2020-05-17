struct TransformNode{M,P} <: Distributions.ContinuousMultivariateDistribution
	m::M
	p::P 
end

Base.length(m::TransformNode) = length(m.p)
Flux.@functor TransformNode


Distributions.logpdf(m::TransformNode, x::AbstractVector) = logpdf(m, reshape(x, :, 1))[1]
function Distributions.logpdf(m::TransformNode, x::AbstractMatrix{T}) where {T}
	x, l = m.m((x,zero(T)))
	logpdf(m.p, x) .+ l[:]
end

function Distributions.logpdf(m::M, x::AbstractMatrix) where {M<: MvNormal{T,Distributions.PDMats.ScalMat{T},FillArrays.Zeros{T,1,Tuple{Base.OneTo{Int64}}}}} where {T}
	log_normal(x, m.μ)[:]
end

function treelogpdf(m::TransformNode, x::AbstractMatrix{T}, tree) where {T}
	x, l = m.m((x,zero(T)))
	treelogpdf(m.p, x, tree) .+ l[:]
end

treecount(m::TransformNode) = treecount(m.p)
sampletree(m::TransformNode) = sampletree(m.p)
_updatelatent!(m::TransformNode, tree) = _updatelatent!(m.p, tree)
zerolatent!(m::TransformNode) = zerolatent!(m.p)
normalizelatent!(m::TransformNode) = normalizelatent!(m.p)

function _maptree(m::TransformNode, x::AbstractMatrix{T}) where {T}
	x, l = m.m((x,zero(T)))
	lkl, tree = _maptree(m.p, x)
	return(lkl .+ l[:], tree)
end


Base.rand(m::TransformNode) = inv(m.m)(rand(m.p))

Base.show(io::IO, m::TransformNode{M,P}) where {M,P} = dsprint(io, m)
function dsprint(io::IO, n::TransformNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → ", color=c)
    dsprint(io, n.p, pad=[pad; (c, "     ")])
end
function dsprint(io::IO, n::TransformNode{M,P}; pad=[]) where {M,P<:MvNormal}
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → MvNormal\n", color=c)
end

