using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumDenseProduct: logsumexp, samplepath, pathlogpdf, batchpathlogpdf
using Flux:throttle

includet("distributions.jl")

function log_likelihoodz(z, components, x)
	lkl = transpose(hcat(map(c -> logpdf(c, x) ,components)...))
	mean(z .* lkl)
end

function estimation_loss(x, proposal_path, logpdfs)
	Flux.mse()
end

x = flower(Float32,200)
K = 9
repetitions = 3
model = SumNode([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K])
qα = Chain(Dense(2,10,relu), Dense(10,10,relu), Dense(10,9)) 
ps = Flux.params(model)
psα = Flux.params(qα)

α₀ = fill(0.001f0, K)

opt = ADAM()
for i in 1:100000
	proposal_path = unique([samplepath(model) for i in 1:repetitions])
	logpdfs = similar(x, size(x,2), length(proposal_path))
	for i in 1:length(proposal_path)
		logpdfs[:,i] .= pathlogpdf(model, x, proposal_path[i])
	end

	idxs = [p[1] for p in proposal_path]
	gs = Flux.gradient(() -> Flux.mse(qα(x)[idxs, :], transpose(logpdfs)), psα)
	Flux.Optimise.update!(opt, psα, gs)

	ρ = qα(x)
	r = ρ ./ sum(ρ, dims = 1)
	α = α₀ + sum(r, dims = 2)[:]

	rt = transpose(r[idxs, :])
	gs = gradient(() -> - sum(rt .* hcat(map(p -> pathlogpdf(model, x, p), proposal_path)...)), ps)

	Flux.Optimise.update!(opt, ps, gs)

	if mod(i, 1000) == 0 
		qloss = Flux.mse(qα(x)[idxs, :], transpose(logpdfs))
		@show qloss
		model.prior .= log.(α)
		@show mean(logpdf(model, x))
		model.prior .= 1
	end
end
