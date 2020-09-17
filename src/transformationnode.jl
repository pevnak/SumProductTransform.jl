struct TransformationNode{M,P}
	m::M
	p::P 
end

Base.length(m::TransformationNode) = length(m.p)
Flux.@functor TransformationNode


function Distributions.logpdf(m::TransformationNode, x::AbstractMatrix{T}) where {T}
	x, l = m.m((x,zero(T)))
	logpdf(m.p, x) .+ l[:]
end

function treelogpdf(m::TransformationNode, x::AbstractMatrix{T}, tree) where {T}
	x, l = m.m((x, zero(T)))
	treelogpdf(m.p, x, tree) .+ l[:]
end

treecount(m::TransformationNode) = treecount(m.p)
sampletree(m::TransformationNode) = sampletree(m.p)
updateprior!(ps::Priors, m::TransformationNode, tree) = updateprior!(ps, m.p, tree)

function _maptree(m::TransformationNode, x::AbstractMatrix{T}) where {T}
	z, l = m.m((x, zero(T)))
	lkl, tree = _maptree(m.p, z)
	return(lkl .+ l[:], tree)
end

# function _maptree(m::TransformationNode, x::AbstractMatrix{T}, s::AbstractScope = NoScope()) where {T}
# 	x, l = m.m((x, zero(T), s))
# 	lkl, tree = _maptree(m.p, x, s)
# 	return(lkl .+ l[:], tree)
# end


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

