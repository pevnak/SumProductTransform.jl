using SumProductTransform:
    TransformationNode,
    SumNode,
    SVDDense,
	MvNormal,
	logpdf,
	logsumexp
using ToyProblems:
	flower2
using Flux:
    gradient,
    params,
	ADAM
using Flux.Optimise:
    update!
using Statistics
using StatsBase:
    Weights,
	sample
using PrayTools:
	classindexes,
	_pgradient
using LatinHypercubeSampling
using Distributions:
	Dirichlet
using Plots
plotly()

function gmm(ncomp::Int)
	SumNode([TransformationNode(SVDDense(2, identity, :butterfly), MvNormal(2, 1f0)) for _ in 1:ncomp])
end

function latinmixture(ncomp::Int, ndata::Int; ndims::Int = 2, scale::Float64 = 5.)
	p = rand(Dirichlet(ncomp, 3.))
	z = sample(1:ncomp, Weights(p), ndata)

	μ = scale * randomLHC(ncomp, ndims)'

	x = μ[:, z] + randn((ndims, ndata))
end

function contours(m::SumNode, x::Array{Float64, 2}, title::String)
	xr = range(minimum(x[1, :]) - 1 , maximum(x[1, :]) + 1 , length = 200)
	yr = range(minimum(x[2, :]) - 1 , maximum(x[2, :]) + 1 , length = 200)

	p = contour(xr, yr, (x...) ->  exp(logpdf(m, [x[1], x[2]])[1]))
	P = scatter!(p, x[1,:], x[2,:], alpha = 0.1)
	title!(p, title)
	p
end

logcond(m, x)  = vcat(map(c -> logpdf(c, x)', m.components)...)
logjoint(m, x) = vcat(map(c -> logpdf(c, x)', m.components)...) .+ m.prior
logjoint(m, x, z) = vcat(map(c -> logpdf(c, x)', m.components[z])...) .+ m.prior[z]

function normlogs(x)
	x = exp.(x .- maximum(x, dims=2))
	r = x ./ sum(x, dims = 1)
end

function normtable(x)
	r = x ./ sum(x, dims = 1)
end

function ∇logpdf(m::SumNode, x::Array{Float64, 2}, ps; multithread::Bool = false)
	if multithread == true
		i = collect(Iterators.partition(1:size(x, 2), Threads.nthreads()))
		samples = [(x[:, j],) for j in i]
		_pgradient((x) -> -sum(logpdf(m, x)), ps, samples)[2]
	else
		gradient(() -> -mean(logpdf(m, x)), ps)
	end
end

function ∇logpdf(m::SumNode, x::Array{Float64, 2}, r::Array{Float64, 2}, ps; multithread::Bool = false)
	if multithread == true
		i = collect(Iterators.partition(1:size(x, 2), Threads.nthreads()))
		samples = [(x[:, j], r[:, j]) for j in i]
		_pgradient((x, r) -> -sum(r .* logjoint(m, x)), ps, samples)[2]
	else
		gradient(() -> -mean(r .* logjoint(m, x)), ps)
	end
end

function ∇logpdf(m::SumNode, x::Array{Float64, 2}, k::Array{Int64, 1}, ps; multithread::Bool = false)
	if multithread == true
		ki = classindexes(k)
		samples = [(x[:, j2], j1) for (j1, j2) in ki]
		_pgradient((x, z) -> -sum(logjoint(m, x, z:z)), ps, samples)[2]
	else
		gradient(() -> -mean(hcat([logjoint(m, x[:, i], k[i:i]) for i in 1:size(x, 2)]...)), ps)
	end
end

function mhsample(m::SumNode, q::Array{Float64, 1}, x::Array{Float64, 1}, pₒ::Float64, kₒ::Int64)
	kₙ = sample(Weights(q))
	pₙ = logpdf(m.components[kₙ], x) + m.prior[kₙ]
	qₒ, qₙ = q[kₒ], q[kₙ]
	accept = log(rand()) < pₙ - pₒ + qₒ - qₙ
	accept ? (pₙ, kₙ) : (pₒ, kₒ)
end

function mhsampler(m::SumNode, q::Array{Float64, 1}, x::Array{Float64, 1}, k::Int64, n::Int64)
	p = logpdf(m.components[k], x) + m.prior[k]
	s = zeros(Int, length(q))
	for i in 1:n
		p, k = mhsample(m, q, x, p, k)
		s[k] += 1
	end
	(k, s)
end

function gd!(m::SumNode, x::Array{Float64, 2}, nstep::Int64, bsize::Int64; opt = ADAM(), ps = params(m), mt::Bool = false)
	for t in 1:nstep
		i = sample(1:size(x, 2), bsize, replace = false)

		gs = ∇logpdf(m, x[:, i], ps, multithread=mt)
        update!(opt, ps, gs)

        mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end

	mean(logpdf(m, x))
end

function em!(m::SumNode, x::Array{Float64, 2}, nstep::Int64, bsize::Int64; opt = ADAM(), ps = params(m), mt::Bool = false)
	for t in 1:nstep
		m.prior .-= logsumexp(m.prior)

		i = sample(1:size(x, 2), bsize, replace = false)
		r = normlogs(logjoint(m, x[:, i]))

		gs = ∇logpdf(m, x[:, i], r, ps, multithread=mt)
		update!(opt, ps, gs)

		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end

	mean(logpdf(m, x))
end

function a1!(m::SumNode, x::Array{Float64, 2}, nstep::Int64, bsize::Int64; opt = ADAM(), ps = params(m), mt::Bool = false)
	for t in 1:nstep
		m.prior .-= logsumexp(m.prior)

		i = sample(1:size(x, 2), bsize, replace = false)
		r = normlogs(logjoint(m, x[:, i]))

		k = [sample(Weights(r[:, j])) for j in 1:bsize]

		gs = ∇logpdf(m, x[:, i], k, ps, multithread=mt)
		update!(opt, ps, gs)

		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end

	mean(logpdf(m, x))
end

function a2!(m::SumNode, x::Array{Float64, 2}, nstep::Int64, bsize::Int64; opt = ADAM(), ps = params(m), mt::Bool = false, nsamp::Int64 = 1, ϕ::Float64 = 0.001)
	ndata = size(x, 2)
	ncomp = length(m.components)
	table = ones(ncomp, ndata)
	k = rand(1:ncomp, ndata)

	for t in 1:nstep
		m.prior .-= logsumexp(m.prior)

		i = sample(1:ndata, bsize, replace = false)
		q = normtable(table[:, i])
		s = zeros(Int8, ncomp, bsize)

		a = 1
		for j in i
			k[j], s[:, a] = mhsampler(m, q[:, a], x[:, j], k[j], nsamp)
			a += 1
		end

		table[:, i] = (1-ϕ)*table[:, i] + ϕ*s
		gs = ∇logpdf(m, x[:, i], k[i], ps, multithread=mt)
		update!(opt, ps, gs)

		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end

	mean(logpdf(m, x))
end


function singlefit()
	ncomp = 5
	ndata = 1000
	bsize = 100
	niter = 10000
	multi = true

	x = flower2(ndata, npetals = 5)
	# x = latinmixture(ncomp, ndata)

	m1 = gmm(ncomp)
	m2 = gmm(ncomp)
	m3 = gmm(ncomp)
	m4 = gmm(ncomp)

	println("__gd__")
	gdt = gd!(m1, x, niter, bsize, mt=multi)
	println("__em__")
	emt = em!(m2, x, niter, bsize, mt=multi)
	println("__a1__")
	a1t = a1!(m3, x, niter, bsize, mt=multi)
	println("__a2__")
	a2t = a2!(m4, x, niter, bsize, mt=multi)

	p1 = contours(m1, x, "Gradient descent $(round.(gdt; digits=2))")
	p2 = contours(m2, x, "Expectation-maximization $(round.(emt; digits=2))")
	p3 = contours(m3, x, "MC score estimator $(round.(a1t; digits=2))")
	p4 = contours(m4, x, "MH score estimator $(round.(a2t; digits=2))")

	plot(p1, p2, p3, p4, layout = (1, 4), size=(1800, 600))
end

function wallclock()
	nruns = 10
	ncomp = 1:2:9
	ndata = 1000
	bsize = 50
	niter = 10000
	nalgs = 4
	multi = true

	lliks = zeros(nruns, length(ncomp), nalgs)
	times = zeros(nruns, length(ncomp), nalgs)
	allcs = zeros(nruns, length(ncomp), nalgs)

	x = flower2(ndata, npetals = 9)
	# x = latinmixture(ncomp, ndata)

	for i in 1:nruns
		for k in 1:length(ncomp)
			println("run # $i, # of components $(ncomp[k])")

			m1 = gmm(ncomp[k])
			m2 = gmm(ncomp[k])
			m3 = gmm(ncomp[k])
			m4 = gmm(ncomp[k])

			println("__gd__")
			gdt = @timed gd!(m1, x, niter, bsize, mt=multi)
			println("__em__")
			emt = @timed em!(m2, x, niter, bsize, mt=multi)
			println("__a1__")
			a1t = @timed a1!(m3, x, niter, bsize, mt=multi)
			println("__a2__")
			a2t = @timed a2!(m4, x, niter, bsize, mt=multi)

			lliks[i, k, 1] = gdt.value
			lliks[i, k, 2] = emt.value
			lliks[i, k, 3] = a1t.value
			lliks[i, k, 4] = a2t.value

			times[i, k, 1] = gdt.time
			times[i, k, 2] = emt.time
			times[i, k, 3] = a1t.time
			times[i, k, 4] = a2t.time

			allcs[i, k, 1] = gdt.bytes
			allcs[i, k, 2] = emt.bytes
			allcs[i, k, 3] = a1t.bytes
			allcs[i, k, 4] = a2t.bytes

			println(" ")
		end
	end

	p1 = plot(
		dropdims(mean(times, dims=1), dims=1),
		dropdims(mean(lliks, dims=1), dims=1),
		xlabel = "computational time [sec]",
		ylabel = "log-likelihood [-]",
		label = ["gd" "em" "a1" "a2"],
		markershape = [:xcross :xcross :xcross :xcross]
	)

	# p2 = plot(
	# 	ncomp,
	# 	dropdims(mean(times, dims=1), dims=1),
	# 	xlabel = "number of components [-]",
	# 	ylabel = "computational time [sec]",
	# 	label = ["gd" "em" "a1" "a2"],
	# 	markershape = [:xcross :xcross :xcross :xcross]
	# )

	# p3 = plot(
	# 	ncomp,
	# 	dropdims(mean(allcs, dims=1), dims=1),
	# 	xlabel = "number of components [-]",
	# 	ylabel = "allocations [bytes]",
	# 	label = ["gd" "em" "a1" "a2"],
	# 	markershape = [:xcross :xcross :xcross :xcross]
	# )

	# plot(p1, p2, p3, layout = (1, 3), size = (1800, 600))
	display(p1)
	# display(p2)
	# display(p3)
end


singlefit()
# wallclock()
