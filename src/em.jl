struct DenseMixture{T,C}
	components::Vector{C}
	prior::Vector{T}
end
Flux.children(x::DenseMixture) = (x.components..., x.prior)
Flux.mapchildren(f, x::DenseMixture) = f.(Flux.children(x))


# Flux.params(m::DenseMixture) = reduce(vcat, [Flux.params(c) for c in m.components])

"""
	estep!(m::MixtureModel, x)
	estep(m::MixtureModel, x)

	updates the prior distribution of components estimated from data `x` given components
	and returns likelihood assuming the most likely component as the generating component
"""
function estep!(m::DenseMixture, x)
	lkl = reduce(vcat, [transpose(logpdf(c, x)) for c in m.components])
	y = mapslices(argmax, softmax(lkl), dims = 1)
	o = Flux.onehotbatch(y[:], 1:length(m.components))
	mean( o .* lkl)
end

function estep(m::DenseMixture, x)
	lkl = hcat([logpdf(c, x) for c in m.components]...)
	y = mapslices(argmax, lkl, dims = 2)
	o = Flux.onehotbatch(y[:], 1:length(m.components))
	mean( transpose(o) .* lkl)
end

function Distributions.logpdf(m::DenseMixture, x::AbstractMatrix)
	lkl = hcat([logpdf(c, x) for c in m.components]...)
	# log.(sum(m.prior' .* exp.(lkl), dims = 2))
	# log.(sum(m.prior' .* exp.(lkl), dims = 2))
	logsumexp(m.prior' .+ lkl, dims = 2)
end

Distributions.logpdf(m::DenseMixture, x::Tuple) = logpdf(m, reshape(collect(x), :, 1))[1]