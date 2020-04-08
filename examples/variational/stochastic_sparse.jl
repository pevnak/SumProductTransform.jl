using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumDenseProduct: logsumexp, samplepath, pathlogpdf, batchpathlogpdf
using Flux:throttle

using Plots
plotly()

function plot_contour(m, x, title = nothing)
	levels = quantile(exp.(logpdf(m, x)), 0.01:0.09:0.99)
	δ = levels[1] / 10
	levels = vcat(collect(levels[1] - 10δ:δ:levels[1] - δ), levels)
	xr = range(minimum(x[1,:]) - 1 , maximum(x[1,:])+ 1 , length = 200)
	yr = range(minimum(x[2,:]) - 1 , maximum(x[2,:])+ 1 , length = 200)
	p1 = contour(xr, yr, (x...) ->  exp(logpdf(m, [x[1],x[2]])[1]))
	p2 = deepcopy(p1)
	xx = x[:,sample(1:size(x,2), 100, replace = false)]
	scatter!(p2, x[1,:], x[2,:], alpha = 0.4)
	p = plot(p1, p2)
	!isnothing(title) && title!(p, title)
	p
end


includet("distributions.jl")

exact = false
VB = false

xx = flower(Float32,2000)
xtst = flower(Float32,2000)
K = 19
repetitions = 6
allpath = [(i, (SumDenseProduct.NoScope(), ())) for i in 1:K]
model = SumNode([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K])
qα = Chain(Dense(2,4*K,relu), Dense(4*K,4*K,relu), Dense(4*K,K)) 
ps = Flux.params(model)
psα = Flux.params(qα)

α₀ = fill(0.001f0, K)
α = deepcopy(α₀)

opt = ADAM()

for i in 1:100000
	global α, exact, VB
	x = xx[:,sample(1:nobs(xx), 200, replace  = false)]
	paths = exact ? allpath : unique([samplepath(model) for i in 1:repetitions]);
	logpdfs = similar(x, length(paths), size(x,2));
	for i in 1:length(paths)
		logpdfs[i, :] .= pathlogpdf(model, x, paths[i])
	end

	idxs = [p[1] for p in paths]
	gs = Flux.gradient(() -> Flux.mse(qα(x)[idxs, :], logpdfs), psα)
	Flux.Optimise.update!(opt, psα, gs)

	ρ = qα(x)
	ρ[idxs,:] .= logpdfs
	ρ .+= VB ? e_dirichlet(α) : log.(α)
	ρ = exp.(ρ .- maximum(ρ,dims=2))
	r = ρ ./ sum(ρ, dims = 1)
	α = α₀ + sum(r, dims = 2)[:]

	rt = transpose(r[idxs, :])
	gs = gradient(ps) do
		- sum(rt .* hcat(map(p -> pathlogpdf(model, x, p), paths)...))
	end

	Flux.Optimise.update!(opt, ps, gs)

	if mod(i, 1000) == 0 
		# error of the 
		y₁ = qα(xx)
		y₂ = vcat(map(c -> Matrix(logpdf(c, xx)'), model.components)...)
		logpdfloss = Flux.mse(y₁, y₂)
		pdfloss = Flux.mse(softmax(y₁), softmax(y₂))

		ρ = vcat(map(c -> Matrix(logpdf(c, xx)'), model.components)...)
	    ρ = exp.(ρ .- maximum(ρ,dims=1))
		r = ρ ./ sum(ρ, dims = 1)
		α = sum(r, dims = 2)[:]
		model.prior .= log.(α)
		exactpdf =  mean(logpdf(model, xtst))

		ρ = qα(xx)
		ρ = exp.(ρ .- maximum(ρ,dims=1))
		r = ρ ./ sum(ρ, dims = 1)
		α = sum(r, dims = 2)[:]
		model.prior .= log.(α)
		variational =  mean(logpdf(model, xtst))

		println(i,":  variational: ",variational,"  exact: ",exactpdf,"  logpdfloss: ", logpdfloss,"  pdfloss: ", pdfloss)
		model.prior .= 1
	end
end
