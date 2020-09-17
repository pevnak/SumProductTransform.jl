struct TransformationNode{M,P}
	m::M
	p::P 
end

Base.length(m::TransformationNode) = length(m.p)
Flux.@functor TransformationNode


Distributions.logpdf(m::TransformationNode, x::AbstractVector) = logpdf(m, reshape(x, :, 1))[1]
Distributions.logpdf(m::TransformationNode, x::AbstractVector, s::NoScope) = logpdf(m, reshape(x, :, 1))[1]
function Distributions.logpdf(m::TransformationNode, x::AbstractMatrix{T}) where {T}
	x, l = m.m((x,zero(T)))
	logpdf(m.p, x) .+ l[:]
end

function Distributions.logpdf(m::TransformationNode, x::AbstractMatrix{T}, s::AbstractScope) where {T}
	x, l, _ = m.m((x, zero(T), s))
	logpdf(m.p, x, s) .+ l[:]
end

function treelogpdf(m::TransformationNode, x::AbstractMatrix{T}, path) where {T}
	s = path[1]
	x, l, _ = m.m((x, zero(T), s))
	treelogpdf(m.p, x, path[2]) .+ l[:]
end

pathcount(m::TransformationNode) = pathcount(m.p)
samplepath(m::TransformationNode) = (NoScope(), samplepath(m.p))
samplepath(m::TransformationNode, s::AbstractScope) = (s, samplepath(m.p, s))
updateprior!(ps::Priors, m::TransformationNode, path) = updateprior!(ps, m.p, path)

function _maptree(m::TransformationNode, x::AbstractMatrix{T}, s::AbstractScope = NoScope()) where {T}
	x, l = m.m((x, zero(T), s))
	lkl, path = _maptree(m.p, x, s)
	return(lkl .+ l[:], path)
end


Base.rand(m::TransformationNode) = inv(m.m)(rand(m.p))

Base.show(io::IO, m::TransformationNode{M,P}) where {M,P} = dsprint(io, m)
function dsprint(io::IO, n::TransformationNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → ", color=c)
    dsprint(io, n.p, pad=[pad; (c, "     ")])
end
function dsprint(io::IO, n::TransformationNode{M,P}; pad=[]) where {M,P<:MvNormal}
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → MvNormal\n", color=c)
end

