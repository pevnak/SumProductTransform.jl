using Flux, ValueHistories
using SumProductTransform: SumNode, logpdf, logsumexp, treelogpdf, sampletree

logjoint(m, x, z) = treelogpdf(m, x, z)
logjoint(m, x, z::Vector) = map(i -> logjoint(m, x[:, i], z[i]), 1:size(x, 2))
logjoint(m, x, z::Matrix) = vcat(map(i -> logjoint(m, x, z[i, :])', 1:size(z, 1))...)

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

function mhsaem!(model, X, batchsize::Int, maxsteps::Int, numsamples::Int; check = 1000, minimum_improvement = typemin(Float64), opt = ADAM(), xval = X, ps = Flux.params(model))
	oldlkl = -mean(logpdf(model, xval))
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
		train_time += @elapsed for j in 1:check
			# E-step
			foreach(p->p.-=logsumexp(p), priors(model))
            b = sample(1:numdata, batchsize, replace = false)
            z[:, b], p[b] = mhsampler(model, X[:, b], z[:, b], p[b], numsamples)
            Q = () -> -mean(logjoint(model, X[:, b], z[:, b]))
            # M-step
            Flux.Optimise.update!(opt, ps, Flux.gradient(Q, ps))
		end
		i += check

		update_time = 0
		likelihood_time = @elapsed newlkl = - mean(batchlogpdf(model, xval, batchsize))
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
