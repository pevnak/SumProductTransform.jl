using Zygote: dropgrad

struct SumNode{T,C}
	components::Vector{C}
	prior::Vector{T}
end

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


Flux.children(x::SumNode) = x.components
Flux.mapchildren(f, x::SumNode) = f.(Flux.children(x))


_priors(m::SumNode) = m.prior

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

	set weight of all components in `m` using data in `x`
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
