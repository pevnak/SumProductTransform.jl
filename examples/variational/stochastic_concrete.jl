using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumDenseProduct: logsumexp
using Flux:throttle

includet("distributions.jl")

x = flower(Float32,200)
K = 9
components = tuple([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K]...)
α₀ = fill(1f0, K)
α = σ.(α₀)
ps = Flux.params(components)
push!(ps, α)
τ = 1f0

q = Chain(Dense(2,10,relu), Dense(2,10,relu), Dense(2,2,softplus))


opt = ADAM()
for i in 1:10000
	global τ, x
	gs = gradient(ps) do 
		α₊ = softplus.(α)
		z = transpose(hard_max(sample_concrete(α₊, τ, size(x, 2)), 1))
		- mean(z .* hcat(map(c -> logpdf(c, x), components)...))
	end

	Flux.Optimise.update!(opt, ps, gs)
	if mod(i, 100) == 0 
		τ /= 2
		@show softplus.(α)
		@show mean(log_likelihood(components, softplus.(α), x))
	end
end

model = SumNode([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K])
fit!(model, x, 100, 10000, 0; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())
