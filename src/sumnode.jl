using Zygote: dropgrad

struct SumNode{T,C}
	components::Vector{C}
	prior::Vector{T}
	function SumNode(components::Vector{C}, prior::Vector{T}) where {T,C}
		ls = length.(components)
		@assert all( ls .== ls[1])
		new{T,C}(components, prior)
	end
end

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

Flux.children(x::SumNode) = x.components
Flux.mapchildren(f, x::SumNode) = f.(Flux.children(x))

"""
	pathlogpdf(p::SumNode, x, path::Vector{Vector{Int}})

	logpdf of samples `x` calculated along the `path`, which determine only 
	subset of models
"""
function pathlogpdf(p::SumNode, x, path) 
	pathlogpdf(p.components[path[1]], x, path[2])
end

"""
	samplepath(m)

	samples path determining subset of the model
"""
function samplepath(m::SumNode) 
	i = rand(1:length(m.components))
	(i, samplepath(m.components[i]))
end

function mappath(m::SumNode, x::AbstractArray{T}) where {T}
	n = length(m.components)
	lkl, path = Vector{Vector{T}}(undef, n), Vector{Any}(undef, n)
	Threads.@threads for i in 1:n
		lkl[i], path[i] = mappath(m.components[i], x)
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


Zygote.@adjoint Flux.onehotbatch(y, n) = Flux.onehotbatch(y,n), Δ -> (nothing, nothing)
Zygote.@adjoint Flux.onecold(x) = Flux.onecold(x), Δ -> (nothing,)


"""
	logpdf(m::MixtureModel, x)

	log-likelihood on samples `x`. During evaluation, weights of mixtures are taken into the account.
	During training, the prior of the sample is one for the most likely component and zero for the others.
"""
function Distributions.logpdf(m::SumNode, x)
	lkl = transpose(hcat(map(c -> logpdf(c, x) ,m.components)...))
	if Flux.istraining()
		y = Flux.onecold(softmax(dropgrad(lkl), dims = 1))
		o = Flux.onehotbatch(y, 1:length(m.components))
		return(mean( o .* lkl, dims = 1)[:])
	else
		return(logsumexp(log.(m.prior .+ 1f-8) .+ lkl, dims = 1)[:])
	end
end


"""
	updatelatent!(m::SumNode, x, bs::Int = typemax(Int))

	estimate the probability of a component in `m` using data in `x`.
	if `bs < size(x,2)`, then the update is calculated part by part to save memory
"""
function updatelatent!(m::SumNode, x, bs::Int = typemax(Int))
	zerolatent!(m);
	foreach(i -> _updatelatent!(m, x[:, i]), Iterators.partition(1:size(x,2),bs))
	normalizelatent!(m);
end

zerolatent!(m) = nothing
function zerolatent!(m::SumNode)
	m.prior .= 0 
	foreach(zerolatent!, m.components)
	nothing
end

normalizelatent!(m) = nothing
function normalizelatent!(m::SumNode)
	m.prior ./= max(sum(m.prior), 1)  
end

function _updatelatent!(m::SumNode, x)
	lkl = transpose(hcat(map(c -> logpdf(c, x),m.components)...))
	y = Flux.onecold(softmax(dropgrad(lkl), dims = 1))
	o = Flux.onehotbatch(y, 1:length(m.components))
	m.prior .+= sum(o, dims = 2)[:]
end
_priors(m::SumNode) = m.prior


Base.show(io::IO, z::SumNode{T,C}) where {T,C} = dsprint(io, z)
function dsprint(io::IO, n::SumNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "Mixture\n", color=c)

    m = length(n.components)
    for i in 1:(m-1)
        paddedprint(io, "  ├── $(n.prior[i])", color=c, pad=pad)
        dsprint(io, n.components[i], pad=[pad; (c, "  │   ")])
    end
    paddedprint(io, "  └── $(n.prior[end])", color=c, pad=pad)
    dsprint(io, n.components[end], pad=[pad; (c, "      ")])
end

