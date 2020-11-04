"""
struct ProductNode
	components::T
	dimensions::U
end

	ProductNode implements a product of independent random variables. Each random 
	variable(s) can be of any type, which implements the interface of `Distributions`
	package (`logpdf` and `length`). Recall that `length` in case of distributions is 
	the dimension of a samples.
"""
struct ProductNode{T<:Tuple,U<:NTuple{N,UnitRange{Int}} where N}
	components::T
	dimensions::U
end

Flux.@functor ProductNode
Flux.trainable(m::ProductNode) = (m.components,)
Base.length(m::ProductNode) = m.dimensions[end].stop
Base.getindex(m::ProductNode, i...) = getindex(m.components, i...)


"""
	ProductNode(ps::Tuple)

	ProductNode with `ps` independent random variables. Each random variable has to 
	implement `logpdf` and `length`.
"""
function ProductNode(ps::Tuple)
	dimensions = Vector{UnitRange{Int}}(undef, length(ps))
	start = 1
	for (i, p) in enumerate(ps)
		l = length(p)
		dimensions[i] = start:start + l - 1
		start += l 
	end
	ProductNode(ps, tuple(dimensions...))
end


####
#	Functions for calculating full likelihood
####
function Distributions.logpdf(m::ProductNode, x)
	m
	o = logpdf(m.components[1], x[m.dimensions[1],:])
	for i in 2:length(m.components)
		o += logpdf(m.components[i], x[m.dimensions[i],:])
	end
	o
end


####
#	Functions supporting calculations of likelihood along trees and their sampling
####
function treelogpdf(p::ProductNode, x, tree)
	o = treelogpdf(p.components[1], x[p.dimensions[1],:], tree[1])
	for i in 2:length(p.components)
		o += treelogpdf(p.components[i], x[p.dimensions[i],:], tree[i])
	end
	o
end
treecount(m::ProductNode) = mapreduce(n -> treecount(n), *, m.components)
sampletree(m::ProductNode) = map(sampletree, m.components)

function _maptree(m::ProductNode, x)
	o, tree = _maptree(m.components[1], x[m.dimensions[1],:],)
	tree = map(s -> (s,), tree)
	for i in 2:length(m.components)
		oo, pp = _maptree(m.components[i], x[m.dimensions[i],:])
		o .+= oo
		tree = map(s -> tuple(s[1]..., s[2]), zip(tree, pp))
	end
	o, tree
end


####
#	Functions for updating prior values by expectation
####
function updateprior!(ps::Priors, m::ProductNode, tree)
	for i in 1:length(m.components)
		updateprior!(ps, m.components[i], tree[i])
	end
end


####
#	Functions for sampling the model
####
Base.rand(m::ProductNode) = vcat([rand(p) for p in m.components]...)


####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:ProductNode}) = InnerNode()
HierarchicalUtils.noderepr(::ProductNode) = "ProductNode"
HierarchicalUtils.printchildren(node::ProductNode) = node.components