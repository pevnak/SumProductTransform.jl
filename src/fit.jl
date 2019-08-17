using IterTools, BSON
"""
	fit!(model, X, batchsize::Int, maxsteps::Int, maxpath::Int; check = 1000, minimum_improvement = 1e-4, opt = ADAM(), debugfile = "", xval = X)

	fits the model using stochastic gradient descend with on data `X` using stochastic
	gradient descend with `batchsize` executed for `maxsteps` with improvement checked every `check` steps
"""
function StatsBase.fit!(model, X, batchsize::Int, maxsteps::Int, maxpath::Int; check = 1000, minimum_improvement = 1e-4, opt = ADAM(), debugfile = "", xval = X)
	ps = Flux.params(model)
	bestpath = [samplepath(model) for i in 1:size(X,2)]
	gradfun = (pathcount(model) < maxpath) ? () -> fullgrad(model, X, batchsize, ps) : () -> samplinggrad(model, X, bestpath, batchsize, maxpath, ps)
	oldlkl = -mean(logpdf(model, xval))
	i = 0;
	train_time = 0.0
	likelihood_time = 0.0
	while true
		train_time += @elapsed for j in 1:check
			gs = gradfun()
			Flux.Optimise.update!(opt, ps, gs)
		end
		i += check
		likelihood_time = @elapsed newlkl = -mean(logpdf(model, xval))
		println(i,": likelihood = ", -newlkl, "  time per iteration: ", train_time / i,"s likelihood time: ",likelihood_time)
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
	updatelatent!(model, X);
	model
end

function samplepdf!(bestpdf, bestpath, model, x, repetitions::Int)
	for i in 2:repetitions
		path = samplepath(model)
		o = pathlogpdf(model, x, path)
		for j in 1:length(o)
			if o[j] > bestpdf[j]
				bestpath[j], bestpdf[j] = path, o[j]
			end
		end
	end
	bestpdf, bestpath
end

function samplinggrad(model, X, bestpath, batchsize, maxpath, ps)
	idxs = sample(1:size(X,2), batchsize, replace = false)
	x = X[:, idxs]
	bestpdf = batchpathlogpdf(model, x, bestpath[idxs])
	bestpdf, bestpath[idxs] = SumDenseProduct.samplepdf!(bestpdf, bestpath[idxs], model, x, maxpath)
	gradient(() -> -mean(batchpathlogpdf(model, x, bestpath[idxs])), ps)
end
	
function fullgrad(model, X, batchsize, ps)
	idxs = sample(1:size(X,2), batchsize, replace = false)
	x = X[:, idxs]
	gradient(() -> -mean(batchpathlogpdf(model, x, bestpath[idxs])), ps)
end
	


function isnaninf(gs)
  for (k,v) in gs.grads
    !isa(v, AbstractArray) && continue
    (any(isinf.(v)) || any(isnan.(v))) && return(true)
  end
  return(false)
end
