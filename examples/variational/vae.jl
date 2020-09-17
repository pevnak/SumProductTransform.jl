using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumDenseProduct: logsumexp
using Flux:throttle

includet("distributions.jl")

x = flower(Float32,200)
K = 9
components = tuple([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K]...)
q = Chain(Dense(2,10,relu), Dense(10,10,relu), Dense(10,9))

ps = Flux.params(components)
push!(ps, Flux.params(q))

@warn "FIXME!!! This formulation misses the KL-Divergence term on multinoulli"
opt = ADAM()
for i in 1:10000
	gs = gradient(ps) do
		- mean(softmax_log_likelihood(components, q(x), x))
	end

	Flux.Optimise.update!(opt, ps, gs)
	if mod(i, 100) == 0 
		@show mean(softmax_log_likelihood(components, q(x), x))
	end
end