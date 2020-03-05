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
	pathlogpdf(p::SumNode, x, path::Vector{Vector{Int}})

	logpdf of samples `x` calculated along the `path`, which determine only 
	subset of models
"""
function pathlogpdf(p::SumNode, x, path, s::AbstractScope = NoScope()) 
	pathlogpdf(p.components[path[1]], x, path[2], s)
end

"""
	samplepath(m)

	samples path determining subset of the model
"""
function samplepath(m::SumNode) 
	i = sample(Weights(softmax(m.prior)))
	(i, samplepath(m.components[i]))
end

function _mappath(m::SumNode, x::AbstractArray{T}, s::AbstractScope) where {T}
	n = length(m.components)
	lkl, path = Vector{Vector{T}}(undef, n), Vector{Any}(undef, n)
	for i in 1:n
		lkl[i], path[i] = _mappath(m.components[i], x, s)
	end
	lkl = transpose(hcat(lkl...))
	y = Flux.onecold(softmax(lkl, dims = 1))
	o = Flux.onehotbatch(y, 1:n)
	o =  sum(o .* lkl, dims = 1)[:]
	path = [(y[i], path[y[i]][i]) for i in 1:size(x,2)]
	return(o, path)
end

pathcount(m::SumNode) = mapreduce(pathcount, +, m.components)

Base.rand(m::SumNode) = rand(m.components[sample(Weights(m.prior))])

"""
	logpdf(m::MixtureModel, x)

	log-likelihood on samples `x`. During evaluation, weights of mixtures are taken into the account.
	During training, the prior of the sample is one for the most likely component and zero for the others.
"""
function Distributions.logpdf(m::SumNode, x, s::AbstractScope = NoScope())
	lkl = transpose(hcat(map(c -> logpdf(c, x, s) ,m.components)...))
	w = softmax(m.prior) .+ 0.001f0
	w = w ./ sum(w)
	logsumexp(log.(w .+ 0.001f0) .+ lkl, dims = 1)[:]
end

function updateprior!(ps::Priors, m::SumNode, path)
	p = get(ps, m.prior, similar(m.prior) .= 0)
	component = path[1]
	p[component] += 1
	updateprior!(ps, m.components[component], path[2])
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

