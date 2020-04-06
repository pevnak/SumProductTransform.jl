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

function log_likelihoodz(z, components, x)
	lkl = transpose(hcat(map(c -> logpdf(c, x) ,components)...))
	mean(z .* lkl)
end

function estimation_loss(x, proposal_path, logpdfs)
	Flux.mse()
end

function updatebestpath!(bestpdf, bestpath, o, path)
	mask = o .> bestpdf
	bestpath[mask] = path[mask]
end


# model = SumNode([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:19])
# fit!(model, x, 100, 20000, 0; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())

x = flower(Float32,200)
K = 19
repetitions = 6
model = SumNode([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K])
qα = Chain(Dense(2,4*K,relu), Dense(4*K,4*K,relu), Dense(4*K,K)) 
ps = Flux.params(model)
psα = Flux.params(qα)

α₀ = fill(0.001f0, K)
α = deepcopy(α₀)

bestpath = [samplepath(model) for i in 1:size(x,2)];

opt = ADAM()
for i in 1:100000
	paths = unique([samplepath(model) for i in 1:repetitions]);
	logpdfs = similar(x, length(paths), size(x,2));
	for i in 1:length(paths)
		logpdfs[i, :] .= pathlogpdf(model, x, paths[i])
	end

	y = mapslices(argmax, logpdfs, dims = 1)
	o = [logpdfs[y[i], i] for i in 1:size(x,2)]
	path = [paths[y[i]] for i in 1:size(x,2)]

	bestlogpdf = batchpathlogpdf(model, x, bestpath)
	updatebestpath!(bestlogpdf, bestpath, o, path)

	idxs = [p[1] for p in paths]
	gs = Flux.gradient(() -> Flux.mse(qα(x)[idxs, :], logpdfs), psα)
	Flux.Optimise.update!(opt, psα, gs)

	ρ = qα(x)
	r = ρ ./ sum(ρ, dims = 1)
	α = α₀ + sum(r, dims = 2)[:]

	rt = transpose(r[idxs, :])
	# rt = logpdfs ./ sum(logpdfs, dims = 2)
	rrt = [r[p[1]] for p in bestpath]
	# gs = gradient(ps) do
	# 	- sum(rt .* hcat(map(p -> pathlogpdf(model, x, p), paths)...))
	# 	-sum(rrt .* batchpathlogpdf(model, x, bestpath))
	# end
	gs = gradient(ps) do
		-sum(batchpathlogpdf(model, x, bestpath))
	end
	Flux.Optimise.update!(opt, ps, gs)

	if mod(i, 1000) == 0 
		y₁ = qα(xx)
		y₂ = vcat(map(c -> Matrix(logpdf(c, xx)'), model.components)...)
		logpdfloss = Flux.mse(y₁, y₂)
		pdfloss = Flux.mse(softmax(y₁), softmax(y₂))

		ρ = vcat(map(c -> Matrix(logpdf(c, x)'), model.components)...)
	    ρ = exp.(ρ .- maximum(ρ,dims=1))
		r = ρ ./ sum(ρ, dims = 1)
		α = sum(r, dims = 2)[:]
		model.prior .= log.(α)
		exact =  mean(logpdf(model, xx))

		ρ = qα(x)
		ρ = exp.(ρ .- maximum(ρ,dims=1))
		r = ρ ./ sum(ρ, dims = 1)
		α = sum(r, dims = 2)[:]
		model.prior .= log.(α)
		variational =  mean(logpdf(model, xx))

		println(i,":  variational: ",variational,"  exact: ",exact,"  logpdfloss: ", logpdfloss,"  pdfloss: ", pdfloss)
	end
end

xx = flower(Float32,1000)
K = 19
repetitions = 6
model = SumNode([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K])
qα = Chain(Dense(2,10,relu), Dense(10,10,relu), Dense(10,K)) 
ps = Flux.params(model)
psα = Flux.params(qα)

α₀ = fill(0.001f0, K)
α = deepcopy(α₀)

opt = ADAM()
for i in 1:100000
	x = xx[:, rand(1:size(xx,2), 100)]
	paths = unique([samplepath(model) for i in 1:repetitions])
	logpdfs = similar(x, size(x,2), length(paths))
	for i in 1:length(paths)
		logpdfs[:,i] .= pathlogpdf(model, x, paths[i])
	end

	idxs = [p[1] for p in paths]
	gs = Flux.gradient(() -> Flux.mse(qα(x)[idxs, :], transpose(logpdfs)), psα)
	Flux.Optimise.update!(opt, psα, gs)

	ρ = qα(x)
	r = ρ ./ sum(ρ, dims = 1)
	α = α₀ + sum(r, dims = 2)[:]

	# rt = transpose(r[idxs, :])
	rt = logpdfs ./ sum(logpdfs, dims = 2)
	gs = gradient(() -> - sum(rt .* hcat(map(p -> pathlogpdf(model, x, p), paths)...)), ps)

	Flux.Optimise.update!(opt, ps, gs)

	if mod(i, 1000) == 0 
		qloss = Flux.mse(qα(xx), vcat(map(c -> Matrix(logpdf(c, xx)'), model.components)...))
		@show qloss
		model.prior .= log.(α)
		@show mean(logpdf(model, xx))
		model.prior .= 1
	end
end

for i in 1:100000
	global α
	paths = unique([samplepath(model) for i in 1:repetitions])
	logpdfs = similar(x, size(x,2), length(paths))
	for i in 1:length(paths)
		logpdfs[:,i] .= pathlogpdf(model, x, paths[i])
	end

	idxs = [p[1] for p in paths]
	gs = Flux.gradient(() -> Flux.mse(qα(x)[idxs, :], transpose(logpdfs)), psα)
	Flux.Optimise.update!(opt, psα, gs)

	ρ = qα(x)
	ρ .+= log.(α)
    ρ = exp.(ρ .- maximum(ρ,dims=2))

	r = ρ ./ sum(ρ, dims = 1)
	α = α₀ + sum(r, dims = 2)[:]

	rt = transpose(r[idxs, :])
	gs = gradient(() -> - sum(rt .* hcat(map(c -> logpdf(c, x), model.components)...)), ps)

	Flux.Optimise.update!(opt, ps, gs)

	if mod(i, 1000) == 0 
		qloss = Flux.mse(qα(x), vcat(map(c -> Matrix(logpdf(c, x)'), model.components)...))
		@show qloss
		model.prior .= log.(α)
		@show mean(logpdf(model, x))
	end
end



x = flower(Float32,200)
K = 9
repetitions = 3
model = SumNode([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K])
qα = Chain(Dense(2,10,relu), Dense(10,10,relu), Dense(10,9)) 
ps = Flux.params(model)
psα = Flux.params(qα)

α₀ = fill(0.001f0, K)
α = deepcopy(α₀)

opt = ADAM()

for i in 1:100000
	global α
	ρ = vcat(map(c -> Matrix(logpdf(c, x)'), model.components)...)

	gs = Flux.gradient(() -> Flux.mse(qα(x), ρ), psα)
	Flux.Optimise.update!(opt, psα, gs)

	ρ = qα(x)
	ρ .+= log.(α)
    ρ = exp.(ρ .- maximum(ρ,dims=2))



	r = ρ ./ sum(ρ, dims = 1)
	α = α₀ + sum(r, dims = 2)[:]

	rt = transpose(r)
	gs = gradient(() -> - sum(rt .* hcat(map(c -> logpdf(c, x), model.components)...)), ps)

	Flux.Optimise.update!(opt, ps, gs)

	if mod(i, 1000) == 0 
		qloss = Flux.mse(qα(x), vcat(map(c -> Matrix(logpdf(c, x)'), model.components)...))
		@show qloss
		model.prior .= log.(α)
		@show mean(logpdf(model, x))
	end
end
