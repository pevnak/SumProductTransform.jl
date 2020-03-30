"""
struct LearnableProductNode
	child::T
	dimensions::U
end

	LearnableProductNode implements a product of independent random variables. Each random 
	variable(s) can be of any type, which implements the interface of `Distributions`
	package (`logpdf` and `length`). Recall that `length` in case of distributions is 
	the dimension of a samples.
"""
struct LearnableProductNode{T<:Number, M}
	α::Matrix{T}
	child::M
end

Flux.@functor LearnableProductNode
# Flux.trainable(m::LearnableProductNode) = (m.child,)

"""
	LearnableProductNode(ps::Tuple)

	LearnableProductNode with `ps` independent random variables. Each random variable has to 
	implement `logpdf` and `length`.
"""
function LearnableProductNode(d::Int, child;max_components = 10)
	LearnableProductNode(randn(Float32, d, min(d, max_components)), child)
end

function Distributions.logpdf(m::LearnableProductNode, x, s::AbstractScope = NoScope())
	#we should add here either exhaustive search or random sampling
	@error "Distributions.logpdf(m::LearnableProductNode, x, s::AbstractScope) is fundamentally broken"
end

function pathlogpdf(p::LearnableProductNode, x, path)
	o = pathlogpdf(p.child, x, path[1][1])
	for i in 2:length(path)
		o += pathlogpdf(p.child, x, path[i][1])
	end
	o
end

pathcount(m::LearnableProductNode) = mapreduce(n -> pathcount(n), *, m.child)

function samplescopes(α, s)
	α = α[active(s), :]
	sind = mapslices(argmax, rand(Float32, size(α)...), dims = 2)
	idxs = collect(1:size(α, 1))
	tuple([s[idxs .== i] for i in unique(sind)]...)
end

function samplepath(m::LearnableProductNode, s::S) where {S<:Union{FullScope, Scope}}
	subscopes = samplescopes(m.α, s)
	map(s -> (s, samplepath(m.child, s)), subscopes)
end

function updateprior!(ps::Priors, m::LearnableProductNode, path)
	for i in 1:length(m.child)
		updateprior!(ps, m.child[i], path[i])
	end
end


function _mappath(m::LearnableProductNode, x, s::AbstractScope = NoScope())
	@error "_mappath(m::LearnableProductNode, x, s::AbstractScope) is fundamentally broken"
end

Base.rand(m::LearnableProductNode) = vcat([rand(p) for p in m.child]...)


Base.length(m::LearnableProductNode) = size(m.α, 1)


Base.show(io::IO, z::LearnableProductNode) = dsprint(io, z)
function dsprint(io::IO, n::LearnableProductNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]

	if typeof(n.child) <: Distributions.MvNormal
	    paddedprint(io, " LearnableProduct → MvNormal\n", color=c)
	else
	    paddedprint(io, " LearnableProduct\n", color=c)
	    paddedprint(io, "  └── ", color=c, pad=pad)
	    dsprint(io, n.child, pad=[pad; (c, "      ")])
	end
end