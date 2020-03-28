using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumDenseProduct: logsumexp
using Flux:throttle

includet("distributions.jl")

function log_likelihoodz(z, components, x)
	lkl = transpose(hcat(map(c -> logpdf(c, x) ,components)...))
	mean(z .* lkl)
end

x = flower(Float32,200)
K = 9
components = tuple([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K]...)
q = Chain(Dense(2,10,relu), Dense(10,10,relu), Dense(10,9))

ps = Flux.params(components)
push!(ps, Flux.params(q))

@warn "FIXME!!! This formulation misses the KL-Divergence term on multinoulli"
opt = ADAM()
τ = 1f0
for i in 1:10000
	global τ
	gs = gradient(ps) do
		α = q(x)

		z = sample_concrete(α, τ)
		- mean(softmax_log_likelihood(components, log.(z), x))
		# z = hard_max(sample_concrete(α, τ), 1)
		# - log_likelihoodz(z, components, x)
	end

	Flux.Optimise.update!(opt, ps, gs)
	if mod(i, 5000) == 0 
		τ /= 2
		@show mean(softmax_log_likelihood(components, q(x), x))
	end
end