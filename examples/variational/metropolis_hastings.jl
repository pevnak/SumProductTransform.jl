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

function a4!(m::SumNode, x::Array{Float64, 2}, nstep::Int; nsamp::Int = 10, opt = ADAM(), ps = params(m), ϕ::Float64 = 0.001)
	ndata = size(x, 2)
	ncomp = length(m.components)

	table = ones(ncomp, ndata)

	for t in 1:nstep
		q = normtable(table)
		z_prev = rand(1:ncomp)

		s = zeros(Int8, ncomp, ndata)
		k = zeros(Int8, nsamp, ndata)
		for i in 1:ndata
			p_prev = logpdf(m.components[z_prev], x[:, i]) + m.prior[z_prev]

			for j in 1:nsamp
				z_curr = sample(Weights(q[:, i]))
				p_curr = logpdf(m.components[z_curr], x[:, i]) + m.prior[z_curr]
				q_curr = log(q[z_curr, i])
				q_prev = log(q[z_prev, i])

				if log(rand()) < p_curr - p_prev + q_prev - q_curr
					p_prev = p_curr
					z_prev = z_curr

					s[z_prev, i] += 1

					k[j, i] = z_curr
				else
					k[j, i] = z_prev
				end
			end
		end

		table = (1-ϕ)*table + ϕ*s

		gs = gradient(() -> -sum(mean(hcat([logjoint(m, x[:, i], k[:, i]) for i in 1:ndata]...), dims=1)), ps)
		update!(opt, ps, gs)

		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end
end

K = 9
N = 100
x = flower2(N, npetals = K)
niter = 100000

# gd!(gmm(K), x, niter)
# println("__________")
# em!(gmm(K), x, niter)
# println("__________")
# a2!(gmm(K), x, niter, nsamp=10)
# println("__________")
# a3!(gmm(K), x, niter)
# println("__________")
a4!(gmm(K), x, niter, nsamp=4)
println("__________")
