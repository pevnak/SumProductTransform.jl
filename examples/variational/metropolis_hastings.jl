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
using PrayTools:
	classindexes,
	_pgradient

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

function ∇logpdf(m, x, k, ps)
	ki = classindexes(k)
	samples = [(m.components[j], x[:, jj], m.prior[j]) for (j, jj) in ki]
	_pgradient((c, x, p) -> -sum(logpdf(c, x) .+ p), ps, samples)
end

function mhsample(m, q, x, pₒ, kₒ)
	kₙ = sample(Weights(q))
	pₙ = logpdf(m.components[kₙ], x) + m.prior[kₙ]
	qₒ, qₙ = q[kₒ], q[kₙ]
	accept = log(rand()) < pₙ - pₒ + qₒ - qₙ
	accept ? (pₙ, kₙ) : (pₒ, kₒ)
end

function mhsampler(m, q, x, k, n)
	p = logpdf(m.components[k], x) + m.prior[k]
	s = zeros(Int, length(q))
	for i in 1:n
		p, k = mhsample(m, q, x, p, k)
		s[k] += 1
	end
	(k, s)
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

function a1!(m::SumNode, x::Array{Float64, 2}, nstep::Int; opt = ADAM(), ps = params(m))
	for t in 1:nstep
		ρ = logjoint(m, x)
		r = normlogs(ρ)

		k = [sample(Weights(r[:, i])) for i in 1:size(x, 2)]

		_, gs = ∇logpdf(m, x, k, ps)
		update!(opt, ps, gs)

		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end
end

function a2!(m::SumNode, x::Array{Float64, 2}, nstep::Int; nsamp::Int = 10, opt = ADAM(), ps = params(m), ϕ::Float64 = 0.001)
	ndata = size(x, 2)
	ncomp = length(m.components)
	table = ones(ncomp, ndata)
	k = rand(1:ncomp, ndata)

	for t in 1:nstep
		q = normtable(table)
		s = zeros(Int8, ncomp, ndata)

		for i in 1:ndata
			k[i], s[:, i] = mhsampler(m, q[:, i], x[:, i], k[i], nsamp)
		end

		table = (1-ϕ)*table + ϕ*s
		_, gs = ∇logpdf(m, x, k, ps)

		update!(opt, ps, gs)
		# updatepriors!(m, x)
		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end
end

K = 9
N = 100
x = flower2(N, npetals = K)
niter = 20000

gd!(gmm(K), x, niter)
println("__________")
em!(gmm(K), x, niter)
println("__________")
a1!(gmm(K), x, niter)
println("__________")
a2!(gmm(K), x, niter, nsamp=1)
println("__________")
