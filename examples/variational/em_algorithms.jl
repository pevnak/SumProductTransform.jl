using SumProductTransform: TransformationNode, SVDDense, logsumexp
using ToyProblems, Flux, Distributions, Unitary, Plots
include("distributions.jl")

function bem(x, comp, α, niter, nepoch, opt)
	α₀, α, c = deepcopy(α), deepcopy(α), deepcopy(comp)
	p = Flux.params(c)

	loglik = zeros(nepoch)

	j = 1

	for t in 1:niter * nepoch
		ρ = logcond(x, c) .+ log.(α)
		r = normlogs(ρ)

		α = α₀ + sum(r, dims = 2)[:]

		g = gradient(() -> - sum(r .* logcond(x, c)), p)
		Flux.Optimise.update!(opt, p, g)

		mod(t, niter) == 0 && (loglik[j] = @show mean(log_likelihood(c, α, x)); j += 1)
	end

	loglik
end

function sem(x, comp, α, niter, nepoch, opt, nk, ni, ϕ)
	α, c = deepcopy(α), deepcopy(comp)
	p = Flux.params(c)

	K = length(c)
	N = size(x, 2)

	ρ = 1e-5*ones(K, N)
	loglik = zeros(nepoch)

	j = 1

	for t in 1:niter * nepoch
		i = randorrange(N, ni)
		k = randorrange(K, nk)

		ck = c[k]

		ρ[k, i] = logcond(x[:, i], ck) .+ log.(α[k])
		r = normlogs(ρ[:, i])

		α = (1-ϕ)*α + ϕ*sum(r, dims=2)[:]

	    g = gradient(() -> - sum(r[k, :] .* logcond(x[:, i], ck)), p)
		Flux.Optimise.update!(opt, p, g)

		mod(t, niter) == 0 && (loglik[j] = @show mean(log_likelihood(c, α, x)); j += 1)
	end

	loglik
end

function sem_cv(x, comp, α, niter, nepoch, opt, nk, ni, ϕ)
	α₀, αt, ct = deepcopy(α), deepcopy(α), deepcopy(comp)
	pt = Flux.params(ct)

	K = length(ct)
	N = size(x, 2)

	ρt = 1e-5*ones(K, N)
	loglik = zeros(nepoch)

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
	    loglik[e] = @show mean(log_likelihood(ct, αt, x))
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


K = 9
N = 200
x = flower(Float32, N)
# m = SumNode([MvNormal(2,1f0) for _ in 1:K])
m = tuple([TransformationNode(SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K]...)
α = fill(0.001f0, K)
opt = ADAM() # step 1e-4

niter = 1000
nepoch = 5
nk = 3
ni = 100
ϕ = 0.005

ll_bem      =    bem(x, m, α, niter, nepoch, opt)
ll_sem_1    =    sem(x, m, α, niter, nepoch, opt,  0,  0, 1.0)
ll_sem_2    =    sem(x, m, α, niter, nepoch, opt,  0,  0, ϕ  )
ll_sem_3    =    sem(x, m, α, niter, nepoch, opt,  0, ni, ϕ  )
ll_sem_4    =    sem(x, m, α, niter, nepoch, opt, nk,  0, ϕ  )
ll_sem_5    =    sem(x, m, α, niter, nepoch, opt, nk, ni, ϕ  )
ll_sem_cv_1 = sem_cv(x, m, α, niter, nepoch, opt,  0,  0, 1.0)
ll_sem_cv_2 = sem_cv(x, m, α, niter, nepoch, opt,  0,  0, ϕ  )
ll_sem_cv_3 = sem_cv(x, m, α, niter, nepoch, opt,  0, ni, ϕ  )
ll_sem_cv_4 = sem_cv(x, m, α, niter, nepoch, opt, nk,  0, ϕ  )
ll_sem_cv_5 = sem_cv(x, m, α, niter, nepoch, opt, nk, ni, ϕ  )

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