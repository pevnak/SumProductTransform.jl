using Flux, ValueHistories, PrayTools
using SumProductTransform: SumNode, logpdf, logsumexp, treelogpdf, sampletree

logjoint(m, x, z) = treelogpdf(m, x, z)
logjoint(m, x, z::Vector) = map(i -> logjoint(m, x[:, i], z[i]), 1:size(x, 2))
logjoint(m, x, z::Matrix) = vcat(map(i -> logjoint(m, x, z[i, :])', 1:size(z, 1))...)

fragment(x, z) = [(x[:, i:i], z[i]) for i in 1:length(z)]
fragment(x, z, n) = [(x[:, i], z[i]) for i in Iterators.partition(1:length(z), div(length(z),n))]


decide(accept, ξₙ, ξₒ) = accept ? ξₙ : ξₒ

function mhsampler(m, x, z, pₒ, numsamples)
    numdata = size(x, 2)
    zₒ = z[end, :]

    for i in 1:numsamples
        zₙ = [sampletree(m) for _ in 1:numdata]
        pₙ = logjoint(m, x, zₙ)
        accept = log.(rand(numdata)) .< pₙ .- pₒ

        pₒ = decide.(accept, pₙ, pₒ)
        zₒ = decide.(accept, zₙ, zₒ)

        z[i, :] = zₒ
    end

    z, pₒ
end

function mhsampler1(m, x, z, pₒ, numsamples)
	@assert size(x,1) == 1
    zₒ = z[end]

    for i in 1:numsamples
        zₙ = sampletree(m)
        pₙ = logjoint(m, x, zₙ)
        accept = log(rand(numdata)) < pₙ - pₒ

        pₒ = decide(accept, pₙ, pₒ)
        zₒ = decide(accept, zₙ, zₒ)

        z[i] = zₒ
    end
    z, pₒ
end

function grad_mhsampler(m, x, z, pₒ, numsamples)
	z, p = Zygote.@ignore mhsampler1(m, x, z, pₒ, numsamples)
	- sum(treelogpdf(model, x, z))
end

function mhsaem!(model, X, batchsize::Int, maxsteps::Int, numsamples::Int; check = 1000, minimum_improvement = typemin(Float64), opt = ADAM(), xval = X, ps = Flux.params(model))
	# oldlkl = -mean(logpdf(model, xval))
	oldlkl = 0
	i = 0;
	train_time = 0.0
	likelihood_time = 0.0
	history = MVHistory()
    numdata = size(X, 2)
    batchsize = min(size(X, 2), batchsize)

    c = map(1:numdata) do _
        [sampletree(model) for _ in 1:numsamples]
    end
    z = hcat(c...)
    p = logjoint(model, X, z[1, :])

	while true
		check = min(check, maxsteps - i)
		# train_time += @elapsed
		for j in 1:check
			# E-step
			foreach(p->p.-=logsumexp(p), priors(model))
            b = sample(1:numdata, batchsize, replace = false)
            z[:, b], p[b] = mhsampler(model, X[:, b], z[:, b], p[b], numsamples)
            # M-step
            xx, zz = X[:, b], z[:, b]
            # gs = PrayTools._pgradient((x...) -> - sum(treelogpdf(model, x...)), ps, fragment(xx, zz))[2]
            gs = PrayTools._pgradient((x...) -> - sum(logjoint(model, x...)), ps, fragment(xx, zz, Threads.nthreads()))[2]
            # gs = Flux.gradient(() -> -mean(logjoint(model,xx,zz)), ps)
            Flux.Optimise.update!(opt, ps, gs)
		end
		i += check

		update_time = 0
		likelihood_time = 0
		# newlkl = - mean(batchlogpdf(model, xval, batchsize))
		newlkl = 0
		println(i,": likelihood = ", -newlkl, "  time per iteration: ", train_time / i,"s update time: ",update_time,"s likelihood time: ",likelihood_time)
		push!(history, :likelihood, i, newlkl)
		push!(history, :traintime, i, train_time)
		push!(history, :likelihoodtime, i, likelihood_time)
		push!(history, :updatelikelihood_time, i, update_time)
		if oldlkl - newlkl < minimum_improvement 
			@info "stopping after $(i) steps due to minimum improvement not met"
			break;
		end
		if i >= maxsteps
			@info "stopping after $(i) steps due to maximum number of steps exceeded"
			break;
		end
		oldlkl = newlkl
	end
	history
end
