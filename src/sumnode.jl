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
	SumNode(components, fill(1/n, n))
end


Base.getindex(m::SumNode,i ::Int) = (c = m.components[i], p = m.prior[i])
Base.length(m::SumNode) = length(m.components[1])

Flux.children(x::SumNode) = x.components
Flux.mapchildren(f, x::SumNode) = f.(Flux.children(x))



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
Distributions.logpdf(m::SumNode, x::Tuple) = logpdf(m, reshape(collect(x), :, 1))[1]


"""
	updatelatent!(m::SumNode, x)

	estimate the probability of a component in `m` using data in `x`
"""
function updatelatent!(m::SumNode, x)
	zerolatent!(m);
	_updatelatent!(m::SumNode, x);
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
    paddedprint(io, "Mixture\n", color=c, pad = pad)

    m = length(n.components)
    for i in 1:(m-1)
        paddedprint(io, "  ├── $(n.prior[i])\n", color=c, pad=pad)
        dsprint(io, n.components[i], pad=[pad; (c, "  │   ")])
    end
    paddedprint(io, "  └── $(n.prior[end])\n", color=c, pad=pad)
    dsprint(io, n.components[end], pad=[pad; (c, "      ")])
end

