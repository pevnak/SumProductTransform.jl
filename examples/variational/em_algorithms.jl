using SumProductTransform: TransformationNode, SVDDense, logsumexp
using ToyProblems, Flux, Distributions, Unitary, Plots
include("distributions.jl")

plotly()


function em!(x, c, α, niter, nepoch; opt=ADAM(), p=Flux.params(c))
	α₀, α = deepcopy(α), deepcopy(α)

	loglik = []

	for t in 1:niter * nepoch
		ρ = logcond(x, c) .+ log.(α)
		r = normlogs(ρ)

		α = α₀ + sum(r, dims = 2)[:]

		g = gradient(() -> - sum(r .* logcond(x, c)), p)
		Flux.Optimise.update!(opt, p, g)

		mod(t, niter) == 0 && (ll = @show mean(log_likelihood(c, α, x)); push!(loglik, ll))
	end

	loglik
end

function sem!(x, c, α, niter, nepoch, nk, ni, ϕ; opt=ADAM(), p=Flux.params(c))
	α = deepcopy(α)

	K = length(c)
	N = size(x, 2)

	ρ = 1e-5*ones(K, N)

	loglik = []

	for t in 1:niter * nepoch
		i = randorrange(N, ni)
		k = randorrange(K, nk)

		ck = c[k]

		ρ[k, i] = logcond(x[:, i], ck) .+ log.(α[k])
		r = normlogs(ρ[:, i])

		α = (1-ϕ)*α + ϕ*sum(r, dims=2)[:]

		g = gradient(() -> - sum(r[k, :] .* logcond(x[:, i], ck)), p)
		Flux.Optimise.update!(opt, p, g)

		mod(t, niter) == 0 && (ll = @show mean(log_likelihood(c, α, x)); push!(loglik, ll))
	end

	loglik
end

function semcv!(x, ct, α, niter, nepoch, nk, ni, ϕ; opt=ADAM(), pt=Flux.params(ct))
	α₀, αt = deepcopy(α), deepcopy(α)

	K = length(ct)
	N = size(x, 2)

	ρt = 1e-5*ones(K, N)

	loglik = []

	for e in 1:nepoch
		ρe = logcond(x, ct) .+ log.(αt)
		re = normlogs(ρe)

		αe = α₀ + sum(re, dims = 2)[:]

		ce = deepcopy(ct)
		pe = Flux.params(ce)
		ge = gradient(() -> - sum(re .* logcond(x, ce)), pe)

		for t in 1:niter
			i = randorrange(N, ni)
			k = randorrange(K, nk)

			ctk = ct[k]
			cek = ce[k]

			ρt[k, i] = logcond(x[:, i], ctk) .+ log.(αt[k])
			rt = normlogs(ρt[:, i])

			αt = (1-ϕ)*αt + ϕ*(sum(rt, dims=2)[:] - sum(re[:, i], dims=2)[:] + αe)

			gti = gradient(() -> - sum(rt[k, :] .* logcond(x[:, i], ctk)), pt)
			gei = gradient(() -> - sum(re[k, i] .* logcond(x[:, i], cek)), pe)

			[gti[a] != nothing && (gti[a] .= gti[a] .- gei[b] .+ ge[b]) for (a, b) in zip(pt, pe)]

	        Flux.Optimise.update!(opt, pt, gti)
		end
	    ll = @show mean(log_likelihood(ct, αt, x)); push!(loglik, ll)
	end

	loglik
end

logcond(x, c) = vcat(map(z -> logpdf(z, x)', c)...)

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

function gmm(kk)
	tuple([TransformationNode(SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:kk]...)
	# SumNode([MvNormal(2,1f0) for _ in 1:K])
end

K = 9
N = 200
x = flower(Float32, N)
α = fill(0.001f0, K)

niter = 1000
nepoch = 20
nk = 3
ni = 100
ϕ = 0.005

ll_bem      =    em!(x, gmm(K), α, niter, nepoch)
ll_sem_1    =   sem!(x, gmm(K), α, niter, nepoch,  0,  0, 1.0)
ll_sem_2    =   sem!(x, gmm(K), α, niter, nepoch,  0,  0, ϕ  )
ll_sem_3    =   sem!(x, gmm(K), α, niter, nepoch,  0, ni, ϕ  )
ll_sem_4    =   sem!(x, gmm(K), α, niter, nepoch, nk,  0, ϕ  )
ll_sem_5    =   sem!(x, gmm(K), α, niter, nepoch, nk, ni, ϕ  )
ll_sem_cv_1 = cvsem!(x, gmm(K), α, niter, nepoch,  0,  0, 1.0)
ll_sem_cv_2 = cvsem!(x, gmm(K), α, niter, nepoch,  0,  0, ϕ  )
ll_sem_cv_3 = cvsem!(x, gmm(K), α, niter, nepoch,  0, ni, ϕ  )
ll_sem_cv_4 = cvsem!(x, gmm(K), α, niter, nepoch, nk,  0, ϕ  )
ll_sem_cv_5 = cvsem!(x, gmm(K), α, niter, nepoch, nk, ni, ϕ  )

p1 = plot(
    hcat(ll_bem, ll_sem_1, ll_sem_2, ll_sem_3, ll_sem_4, ll_sem_5),
    label=["bem" "sem-sanity check" "sem-none" "sem-x" "sem-z" "sem-xz"],
    markershape=[:circle :xcross :cross :circle :circle :circle :circle],
    xlabel="epoch [-]",
    ylabel="log-likelihood [-]",
    title="sEM",
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend=:bottomright,
    )

p2 = plot(
    hcat(ll_bem, ll_sem_cv_1, ll_sem_cv_2, ll_sem_cv_3, ll_sem_cv_4, ll_sem_cv_5),
    label=["bem" "sem_cv-sanity check" "sem_cv-none" "sem_cv-x" "sem_cv-z" "sem_cv-xz"],
    markershape=[:circle :xcross :cross :circle :circle :circle],
    xlabel="epoch [-]",
    ylabel="log-likelihood [-]",
    title="sEM control variates",
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend=:bottomright,
    )

l = @layout [a b]
plot(p1, p2, layout = l)
