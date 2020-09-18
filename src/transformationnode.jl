struct TransformationNode{M,P}
	m::M
	p::P 
end

Base.length(m::TransformationNode) = length(m.p)
Flux.@functor TransformationNode

####
#	Functions for calculating full likelihood
####
function Distributions.logpdf(m::TransformationNode, x::AbstractMatrix{T}) where {T}
	x, l = m.m((x,zero(T)))
	logpdf(m.p, x) .+ l[:]
end


####
#	Functions supporting calculations of likelihood along trees and their sampling
####
function treelogpdf(m::TransformationNode, x::AbstractMatrix{T}, tree) where {T}
	x, l = m.m((x, zero(T)))
	treelogpdf(m.p, x, tree) .+ l[:]
end

treecount(m::TransformationNode) = treecount(m.p)

sampletree(m::TransformationNode) = sampletree(m.p)

function _maptree(m::TransformationNode, x::AbstractMatrix{T}) where {T}
	z, l = m.m((x, zero(T)))
	lkl, tree = _maptree(m.p, z)
	return(lkl .+ l[:], tree)
end


####
#	Functions for updating prior values by expectation
####
updateprior!(ps::Priors, m::TransformationNode, tree) = updateprior!(ps, m.p, tree)


####
#	Functions for sampling the model
####
Base.rand(m::TransformationNode) = inv(m.m)(rand(m.p))


####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:TransformationNode}) = InnerNode()
HierarchicalUtils.noderepr(node::TransformationNode) = "$(node.m) →"
HierarchicalUtils.printchildren(node::TransformationNode) = (node.p,)

HierarchicalUtils.NodeType(::Type{TransformationNode{T,P}}) where {T,P<:MvNormal} = LeafNode()
HierarchicalUtils.noderepr(node::TransformationNode{T,P})  where {T,P<:MvNormal} = "$(node.m) → MvNormal"
HierarchicalUtils.printchildren(node::TransformationNode{T,P}) where {T,P<:MvNormal} = tuple()
