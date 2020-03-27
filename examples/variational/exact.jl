using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumDenseProduct: logsumexp
using Flux:throttle
include("distributions.jl")


x = flower(Float32,200)
K = 9
components = tuple([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K]...)
α₀ = fill(1f0, K)
α = deepcopy(α₀)
ps = Flux.params(components)
opt = ADAM()

for i in 1:20000
	global α, components, opt
	ρ = vcat(map(c -> Matrix(logpdf(c, x)'), components)...)
	# ρ .+= e_dirichlet(α)
	r = ρ ./ sum(ρ, dims = 1)
	α = α₀ + sum(r, dims = 2)[:]
	# ed = e_dirichlet(α)

	r = transpose(r)
	gs = gradient(() -> - sum(r .* hcat(map(c -> logpdf(c, x), components)...)), ps)
	# gs = gradient(() -> - sum(logsumexp(ed .+ r .+ hcat(map(c -> logpdf(c, x), components)...), dims = 2)), ps)
	Flux.Optimise.update!(opt, ps, gs)
	mod(i, 1000) == 0 && @show mean(log_likelihood(components, α, x))
end
# model = SumNode([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K])
# fit!(model, x, 100, 10000, 0; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())
