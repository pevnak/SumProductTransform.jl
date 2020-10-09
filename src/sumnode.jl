using StatsBase


struct SumNode{T,C}
	components::Vector{C}
	prior::Vector{T}
	function SumNode(components::Vector{C}, prior::Vector{T}) where {T,C}
		ls = length.(components)
		@assert all( ls .== ls[1])
		new{T,C}(components, prior)
	end
end

_priors(m::SumNode) = m.prior

"""
	SumNode(components::Vector, prior::Vector)
	SumNode(components::Vector) 

	Mixture of components. Each component has to be a valid pdf. If prior vector 
	is not provided, it is initialized to uniform.
"""
function SumNode(components::Vector) 
	n = length(components); 
	SumNode(components, fill(1f0, n))
end

Base.getindex(m::SumNode, i ::Int) = (c = m.components[i], p = m.prior[i])
Base.length(m::SumNode) = length(m.components[1])

Flux.@functor SumNode


####
#	Functions for calculating full likelihood
####
"""
	logpdf(node, x)

	log-likelihood of samples `x` of a model `node`
"""
function Distributions.logpdf(m::SumNode, x::AbstractMatrix)
	lkl = transpose(hcat(map(c -> logpdf(c, x) ,m.components)...))
	w = m.prior .- logsumexp(m.prior)
	logsumexp(w .+ lkl, dims = 1)[:]
end


####
#	Functions supporting calculations of likelihood along trees and their sampling
####
"""
	treelogpdf(node, x, tree)

	logpdf of samples `x` calculated along the `tree`, which determine only  subset of models
"""
function treelogpdf(p::SumNode, x, tree) 
	i = tree[1]
	w = p.prior .- logsumexp(p.prior)
	w[i] .+ treelogpdf(p.components[i], x, tree[2])
end

"""
	sampletree(m)

	randomly sample a `tree` from the SPTN model
"""
function sampletree(m::SumNode) 
	i = sample(Weights(softmax(m.prior)))
	(i, sampletree(m.components[i]))
end

"""
	(likelihood, tree) = _maptree(m, x)

	`likelihood` of the most probable `tree` for a model `m` of sample `x` 
"""
function _maptree(m::SumNode, x::AbstractArray{T}) where {T}
	n = length(m.components)
	lkl, tree = _maptree(m.components[1], x)
	tree = map(t -> (1, t), tree)
	for i in 2:n
		_lkl, _tree = _maptree(m.components[i], x)
		(lkl, tree) = keepbetter((lkl, tree), (_lkl, _tree), i)
	end
	return(lkl, tree)
end

function keepbetter(a, b, i)
	better =  (a[1] .> b[1])
	lkl = [better[j] ? a[1][j] : b[1][j] for j in 1:length(better)]
	tree = [better[j] ? a[2][j] : (i, b[2][j]) for j in 1:length(better)]
	(lkl, tree)
end

treecount(m::SumNode) = mapreduce(treecount, +, m.components)


####
#	Functions for updating prior values by expectation
####
function updateprior!(ps::Priors, m::SumNode, tree)
	p = get(ps, m.prior, similar(m.prior) .= 0)
	component = tree[1]
	p[component] += 1
	updateprior!(ps, m.components[component], tree[2])
end


####
#	Functions for sampling the model
####
Base.rand(m::SumNode) = rand(m.components[sample(Weights(m.prior))])


####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.noderepr(node::SumNode) = "SumNode"
HierarchicalUtils.NodeType(::Type{<:SumNode}) = InnerNode()
# HierarchicalUtils.printchildren(node::SumNode) = [Symbol(w) => ch for (w, ch) in zip(node.prior, node.components)]
HierarchicalUtils.printchildren(node::SumNode) = tuple(node.components...)

