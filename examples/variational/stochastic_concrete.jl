using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumDenseProduct: logsumexp
using Flux:throttle

includet("distributions.jl")


function log_likelihood(α, components, x)
	mean(logsumexp(log.(α) .+  hcat(map(c -> logpdf(c, x), components)...), dims = 1))
end

function log_likelihoodz(z, components, x)
	mean(z .* hcat(map(c -> logpdf(c, x), components)...))
end

x = flower(Float32,200)
K = 9
components = tuple([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K]...)
α₀ = fill(1f0, K)
α = reshape(σ.(α₀), : ,1)
α = σ.(α₀)
ps = Flux.params(components)
push!(ps, α)
τ = 1f0

opt = ADAM()
for i in 1:10000
	global τ, x
	gs = gradient(ps) do 
		# α₊ = softplus.(α)
		z = transpose(hard_max(sample_concrete(α, τ, size(x, 2)), 1))
		- log_likelihoodz(z, components, x)
		# - mean(z .* hcat(map(c -> logpdf(c, x), components)...))
		# α₊ = transpose(softmax(α))
		# - log_likelihood(α₊, components, x)
		# - log_likelihood(α₊, components, x)
	end

	Flux.Optimise.update!(opt, ps, gs)
	if mod(i, 100) == 0 
		τ /= 2
		# @show mean(log_likelihood(components, softplus.(α), x))
		@show log_likelihood(softmax(softplus.(α))', components, x)
	end
end

model = SumNode([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K])
fit!(model, x, 100, 10000, 0; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())

log_likelihood(softmax(model.prior)', model.components, x)