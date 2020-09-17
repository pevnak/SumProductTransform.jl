using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumDenseProduct: logsumexp
using Flux:throttle

includet("distributions.jl")

x = flower(Float32,200)
K = 9
components = tuple([TransformationNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K]...)
ps = Flux.params(components)
τ = 1f0

q = Chain(Dense(2,10,relu), Dense(10,10,relu), Dense(10,9))
push!(ps, Flux.params(q))

opt = ADAM()
for i in 1:10000
	global τ, x
	gs = gradient(ps) do
		# α₊ = q(x)
		w = m.prior .- logsumexp(m.prior)
		logsumexp(w .+ lkl, dims = 1)[:]
 
		# z = transpose(hard_max(sample_concrete(α₊, τ), 1))
		# z = transpose(sample_concrete(α₊, τ))
		- mean(z .* hcat(map(c -> logpdf(c, x), components)...))
	end

	Flux.Optimise.update!(opt, ps, gs)
	if mod(i, 100) == 0 
		τ /= 2
		@show mean(log_likelihood(components, softplus.(α), x))
	end
end

model = SumNode([TransformationNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K])
fit!(model, x, 100, 10000, 0; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())
