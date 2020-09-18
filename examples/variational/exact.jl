using ToyProblems, Distributions, SumProductTransform, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumProductTransform: logsumexp
using Flux:throttle
include("distributions.jl")


x = flower(Float32,200);
K = 19
comps = tuple([TransformationNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K]...)
α₀ = fill(0.001f0, K)
α = deepcopy(α₀)
ps = Flux.params(comps)
opt = ADAM()


VB = true;
niter = 20000;
for i in 1:niter
	global α, comps, opt

	ρ = vcat(map(c -> Matrix(logpdf(c, x)'), comps)...)
	if VB
		ρ .+=  e_dirichlet(α)
	else
		ρ .+= log.(α)
	end
    ρ = exp.(ρ .- maximum(ρ,dims=2))

	# ρ .+= 
	r = ρ ./ sum(ρ, dims = 1)
	α = α₀ + sum(r, dims = 2)[:]
	# ed = e_dirichlet(α)

	r = transpose(r)
	gs = gradient(() -> - sum(r .* hcat(map(c -> logpdf(c, x), comps)...)), ps)
	# gs = gradient(() -> - sum(logsumexp(ed .+ r .+ hcat(map(c -> logpdf(c, x), comps)...), dims = 2)), ps)
	Flux.Optimise.update!(opt, ps, gs)
	mod(i, 1000) == 0 && @show mean(log_likelihood(comps, α, x)), minimum(α)
end

model = SumNode(collect(comps))
fit!(model, x, 100, 10000, 0; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())
