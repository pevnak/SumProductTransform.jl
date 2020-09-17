using IterTools, BSON, ValueHistories, MLDataPattern

samplebatch(x, bs) = x[:,sample(1:size(x,2), min(size(x,2),bs), replace = false)]
"""
	fit!(model, X, batchsize::Int, maxsteps::Int, maxpath::Int; check = 1000, minimum_improvement = typemin(Float64), opt = ADAM(), debugfile = "", xval = X)

	fits the model using stochastic gradient descend with on data `X` using stochastic
	gradient descend with `batchsize` executed for `maxsteps` with improvement checked every `check` steps
"""
function em!(model, X, batchsize::Int, maxsteps::Int; check = 1000, minimum_improvement = typemin(Float64), opt = ADAM(), debugfile = "", xval = X, msteps = 10)
	ps = Flux.params(model)
	oldlkl = -mean(logpdf(model, xval))
	i = 0;
	history = MVHistory()
	start_time = time()
	while true
		check = min(check, maxsteps - i)
		for j in 1:check
			for _ in 1:msteps
				@timeit to "mstep!" mstep!(model, ps, opt, X, batchsize)
			end
			@timeit to "updatepriors!" updatepriors!(model, samplebatch(X, 10*batchsize));
			# println("$(j): $(start_time / ((i+j)))")
		end
		i += check
		newlkl = @timeit to "validation" (-mean(batchlogpdf(model, xval, batchsize)))
		println("$(i): $(-newlkl)")
		display(SumDenseProduct.to)
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

function mstep!(model, ps, opt, x, batchsize)
	xx = @timeit to "samplebatch" samplebatch(x, batchsize)
	paths = @timeit to "samplepath" [samplepath(model) for i in 1:batchsize]
	gs = @timeit to "gradient" threadedgrad(i -> -sum(batchtreelogpdf(model, xx[:,i], paths[i])), ps, size(xx,2))
	# gs = @timeit to "gradient" gradient(() -> -sum(batchtreelogpdf(model, xx, paths)), ps)
	@timeit to "update!" Flux.Optimise.update!(opt, ps, gs)
end
