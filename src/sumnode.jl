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
	SumNode(components, fill(Float32(1/n), n))
end

Base.getindex(m::SumNode,i ::Int) = (c = m.components[i], p = m.prior[i])
Base.length(m::SumNode) = length(m.components[1])

Flux.@functor SumNode

"""
	treelogpdf(p::SumNode, x, tree::Vector{Vector{Int}})

	logpdf of samples `x` calculated along the `tree`, which determine only 
	subset of models
"""
function treelogpdf(p::SumNode, x, tree) 
	treelogpdf(p.components[tree[1]], x, tree[2])
end

"""
	sampletree(m)

	samples tree determining subset of the model
"""
function sampletree(m::SumNode, s...) 
	i = sample(Weights(softmax(m.prior)))
	(i, sampletree(m.components[i], s...))
end

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

Base.rand(m::SumNode) = rand(m.components[sample(Weights(m.prior))])

"""
	logpdf(m::MixtureModel, x)

	log-likelihood on samples `x`. During evaluation, weights of mixtures are taken into the account.
	During training, the prior of the sample is one for the most likely component and zero for the others.
"""

function Distributions.logpdf(m::SumNode, x::AbstractMatrix)
	lkl = transpose(hcat(map(c -> logpdf(c, x) ,m.components)...))
	w = m.prior .- logsumexp(m.prior)
	logsumexp(w .+ lkl, dims = 1)[:]
end

function updateprior!(ps::Priors, m::SumNode, tree)
	p = get(ps, m.prior, similar(m.prior) .= 0)
	component = tree[1]
	p[component] += 1
	updateprior!(ps, m.components[component], tree[2])
end

Base.show(io::IO, z::SumNode{T,C}) where {T,C} = dsprint(io, z)
function dsprint(io::IO, n::SumNode; pad=[])
	w = softmax(n.prior) .+ 0.001f0
	w = w ./ sum(w)
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "Mixture\n", color=c)

    m = length(n.components)
    for i in 1:(m-1)
        paddedprint(io, "  ├── $(w[i])", color=c, pad=pad)
        dsprint(io, n.components[i], pad=[pad; (c, "  │   ")])
    end
    paddedprint(io, "  └── $(w[end])", color=c, pad=pad)
    dsprint(io, n.components[end], pad=[pad; (c, "      ")])
end

