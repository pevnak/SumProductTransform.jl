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
using Plots

plotly()

function gmm(k)
	SumNode([TransformationNode(SVDDense(2, identity, :butterfly), MvNormal(2, 1f0)) for _ in 1:k])
end

logcond(m, x)  = vcat(map(c -> logpdf(c, x)', m.components)...)
logjoint(m, x) = vcat(map(c -> logpdf(c, x)', m.components)...) .+ m.prior
logjoint(m, x, z) = vcat(map(c -> logpdf(c, x)', m.components[z])...) .+ m.prior[z]

function normlogs(x)
	x = exp.(x .- maximum(x, dims=2))
	r = x ./ sum(x, dims = 1)
end

function randorrange(m, n)
	if n == 0
		i = 1:m
	else
		i = rand(1:m, n)
	end
end

function em!(m::SumNode, x::Array{Float64, 2}, niter::Int, nepoch::Int; opt = ADAM(), ps = params(m))
	loglik = []

	for t in 1:niter * nepoch
		ρ = logjoint(m, x)
		r = normlogs(ρ)

		gs = gradient(() -> -sum(r .* logjoint(m, x)), ps)
		update!(opt, ps, gs)

		mod(t, niter) == 0 && (ll = @show mean(logpdf(m, x)); push!(loglik, ll))
	end
end

function sem!(m::SumNode, x::Array{Float64, 2}, niter::Int, nepoch::Int, nk::Int, ni::Int; opt = ADAM(), ps = params(m))
	K = size(m.prior, 1)
	N = size(x, 2)

	ρ = 1e-5*ones(K, N)

	loglik = []

	for t in 1:niter * nepoch
		i = randorrange(N, ni)
		k = randorrange(K, nk)

		ρ[k, i] = logjoint(m, x[:, i], k)
		r = normlogs(ρ[:, i])

		gs = gradient(() -> - sum(r[k, :] .* logjoint(m, x[:, i], k)), ps)
		update!(opt, ps, gs)

		mod(t, niter) == 0 && (ll = @show mean(logpdf(m, x)); push!(loglik, ll))
	end

	loglik
end

function semcv!(mt::SumNode, x::Array{Float64, 2}, niter::Int, nepoch::Int, nk::Int, ni::Int; opt = ADAM(), pt = params(mt))
	K = size(mt.prior, 1)
	N = size(x, 2)

	ρt = 1e-5*ones(K, N)

	loglik = []

	for e in 1:nepoch
		ρe = logjoint(mt, x)
		re = normlogs(ρe)

		me = deepcopy(mt)
		pe = params(me)
		ge = gradient(() -> - sum(re .* logjoint(me, x)), pe)

		for t in 1:niter
			i = randorrange(N, ni)
			k = randorrange(K, nk)

			ρt[k, i] = logjoint(mt, x[:, i], k)
			rt = normlogs(ρt[:, i])

			gti = gradient(() -> - sum(rt[k, :] .* logjoint(mt, x[:, i], k)), pt)
			gei = gradient(() -> - sum(re[k, i] .* logjoint(me, x[:, i], k)), pe)

			[gti[a] !== nothing && (gti[a] .= gti[a] .- gei[b] .+ ge[b]) for (a, b) in zip(pt, pe)]

	        update!(opt, pt, gti)
		end
	    ll = @show mean(logpdf(mt, x)); push!(loglik, ll)
	end

	loglik
end

K = 9
N = 200
x = flower2(N, npetals = K)

niter = 1000
nepoch = 20
nk = 3
ni = 100

ll_bem      =    em!(gmm(K), x, niter, nepoch)
ll_sem_1    =   sem!(gmm(K), x, niter, nepoch,  0,  0) # ϕ=1.0
# ll_sem_2    =   sem!(x, gmm(K), α, niter, nepoch,  0,  0, ϕ  )
# ll_sem_3    =   sem!(x, gmm(K), α, niter, nepoch,  0, ni, ϕ  )
# ll_sem_4    =   sem!(x, gmm(K), α, niter, nepoch, nk,  0, ϕ  )
# ll_sem_5    =   sem!(x, gmm(K), α, niter, nepoch, nk, ni, ϕ  )
ll_sem_cv_1 = semcv!(gmm(K), x, niter, nepoch,  0,  0) # ϕ=1.0
# ll_sem_cv_2 = semcv!(x, gmm(K), α, niter, nepoch,  0,  0, ϕ  )
# ll_sem_cv_3 = semcv!(x, gmm(K), α, niter, nepoch,  0, ni, ϕ  )
# ll_sem_cv_4 = semcv!(x, gmm(K), α, niter, nepoch, nk,  0, ϕ  )
# ll_sem_cv_5 = semcv!(x, gmm(K), α, niter, nepoch, nk, ni, ϕ  )

# p1 = plot(
#     hcat(ll_bem, ll_sem_1, ll_sem_2, ll_sem_3, ll_sem_4, ll_sem_5),
#     label=["bem" "sem-sanity check" "sem-none" "sem-x" "sem-z" "sem-xz"],
#     markershape=[:circle :xcross :cross :circle :circle :circle :circle],
#     xlabel="epoch [-]",
#     ylabel="log-likelihood [-]",
#     title="sEM",
#     foreground_color_legend = nothing,
#     background_color_legend = nothing,
#     legend=:bottomright,
#     )

# p2 = plot(
#     hcat(ll_bem, ll_sem_cv_1, ll_sem_cv_2, ll_sem_cv_3, ll_sem_cv_4, ll_sem_cv_5),
#     label=["bem" "sem_cv-sanity check" "sem_cv-none" "sem_cv-x" "sem_cv-z" "sem_cv-xz"],
#     markershape=[:circle :xcross :cross :circle :circle :circle],
#     xlabel="epoch [-]",
#     ylabel="log-likelihood [-]",
#     title="sEM control variates",
#     foreground_color_legend = nothing,
#     background_color_legend = nothing,
#     legend=:bottomright,
#     )

# l = @layout [a b]
# plot(p1, p2, layout = l)
