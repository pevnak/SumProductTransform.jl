using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumDenseProduct: logsumexp
using Flux:throttle

include("distributions.jl")

x = flower(Float32,200)
K = 9
components = tuple([TransformationNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K]...)
α₀ = fill(1f0, K)
α = σ.(α₀)
ps = Flux.params(components)
push!(ps, α)


opt = ADAM()
for i in 1:10000
	α₊ = softplus.(α)
	z = samplez(α₊, size(x,2))
	gs = gradient(() -> - mean(z .* hcat(map(c -> logpdf(c, x), components)...)), ps)
	∇α₊ = - mean(z .* hcat(map(c -> logpdf(c, x), components)...) .* ∇log_samplezi(α₊, z), dims = 1)[:]
	∇α₊ = ∇α₊ + gradient(x -> kldir(x, α₀), α₊)[1]

	gs.grads[α] = σ.(α₊) .* ∇α₊
	Flux.Optimise.update!(opt, ps, gs)
	if mod(i, 100) == 0 
		@show α₊
		@show mean(log_likelihood(components, α₊, x))
	end
end

model = SumNode([TransformationNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K])
fit!(model, x, 100, 10000, 0; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())
