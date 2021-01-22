using SumProductTransform:
    TransformationNode,
    SumNode,
    SVDDense,
	MvNormal,
	logpdf,
	logsumexp,
	updatepriors!
using ToyProblems:
	flower2
using Flux:
    gradient,
    params,
	ADAM,
	onehotbatch
using Flux.Optimise:
    update!
using Statistics
using StatsBase:
    Weights,
    sample

function gmm(k)
	SumNode([TransformationNode(SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:k])
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

function gd!(m::SumNode, x::Array{Float64, 2}, nstep::Int, opt = ADAM(), ps = params(m))
    for t in 1:nstep
        gs = gradient(() -> -mean(logpdf(m, x)), ps)
        update!(opt, ps, gs)

        mod(t, 1000) == 0 && @show mean(logpdf(m, x))
    end
end

function em!(m::SumNode, x::Array{Float64, 2}, nstep::Int, opt = ADAM(), ps = params(m))
	for t in 1:nstep
		ρ = logjoint(m, x)
		r = normlogs(ρ)

		gs = gradient(() -> -sum(r .* logjoint(m, x)), ps)
		update!(opt, ps, gs)

		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end
end

function a2!(m::SumNode, x::Array{Float64, 2}, nstep::Int; nsamp::Int = 10, opt = ADAM(), ps = params(m))
	for t in 1:nstep
		ρ = logjoint(m, x)
		r = normlogs(ρ)

		k = hcat([[sample(Weights(r[:, i])) for _ in 1:nsamp] for i in 1:size(x, 2)]...)

		gs = gradient(() -> -sum(mean(hcat([logjoint(m, x[:, i], k[:, i]) for i in 1:size(x, 2)]...), dims=1)), ps)
		update!(opt, ps, gs)

		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end
end

function a3!(m::SumNode, x::Array{Float64, 2}, nstep::Int; nsamp::Int = 10, opt = ADAM(), ps = params(m))
	for t in 1:nstep
		ρ = sum(logcond(m, x), dims=2) + m.prior
		r = normlogs(ρ)

		k = [sample(Weights(r[:])) for _ in 1:nsamp]

		gs = gradient(() -> -sum(mean(hcat([logjoint(m, x[:, i], k[:]) for i in 1:size(x, 2)]...), dims=1)), ps)
		update!(opt, ps, gs)

		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end
end

function mhsample(components, q, x, pₒ, kₒ)
	kₙ = sample(Weights(q))
	pₙ = logpdf(components[kₙ], x)
	qₒ, qₙ = q[kₒ], q[kₙ]
	accept = log(rand()) < pₙ - pₒ + qₒ - qₙ
	accept ? (pₙ, kₙ) : (pₒ, kₒ)
end

function mhsampler(components, q, x, k, n)
	p = logpdf(components[k], x)
	s = zeros(Int, length(q))
	for i in 1:n
		p, k = mhsample(components, q, x, p, k)
		s[k] += 1
	end
	(p, k, s)
end

using PrayTools

function ∇tlogpdf(components, x, k, ps)
	ki = PrayTools.classindexes(k)
	samples = [(components[j], x[:, jj]) for (j,jj) in ki]
	PrayTools._pgradient((c, x) -> -sum(logpdf(c, x)), ps, samples)
end

function a4!(m::SumNode, x::Array{Float64, 2}, nstep::Int; nsamp::Int = 10, opt = ADAM(), ps = params(m), ϕ::Float64 = 0.001)
	ndata = size(x, 2)
	ncomp = length(m.components)
	table = ones(ncomp, ndata)
	k = rand(1:ncomp, ndata)

	for t in 1:nstep
		q = normtable(table)
		ii = rand(1:ndata, 100)
		s = zeros(Int8, ncomp, length(ii))

		for (j,i) in enumerate(ii)
			_, k[i], s[:, j] = mhsampler(m.components, q[:, i], x[:,i], k[i], nsamp)
		end

		table[:,ii] = (1-ϕ)*table[:,ii] + ϕ*s
		# gs = gradient(() -> -sum(mean(hcat([logjoint(m, x[:, i], z[:, i]) for i in 1:ndata]...), dims=1)), ps)
		# gs = gradient(() -> -sum(logpdf(m.components[k[i]], x[:, i]) for i in 1:ndata), ps)
		lp, gs = ∇tlogpdf(m.components, x[:,ii], k[ii], ps)
		
		update!(opt, ps, gs)
		updatepriors!(m, x)
		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end
end

K = 9
N = 10000
x = flower2(N, npetals = K)
niter = 100000

gd!(gmm(K), x, niter)
# println("__________")
# em!(gmm(K), x, niter)
# println("__________")
# a2!(gmm(K), x, niter, nsamp=10)
# println("__________")
# a3!(gmm(K), x, niter)
# println("__________")
# m = gmm(K)
a4!(gmm(K)	, x, niter, nsamp=4)
println("__________")
