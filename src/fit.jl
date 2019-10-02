using IterTools, BSON, ValueHistories
using TimerOutputs
const to = TimerOutput()
"""
	fit!(model, X, batchsize::Int, maxsteps::Int, maxpath::Int; check = 1000, minimum_improvement = typemin(Float64), opt = ADAM(), debugfile = "", xval = X)

	fits the model using stochastic gradient descend with on data `X` using stochastic
	gradient descend with `batchsize` executed for `maxsteps` with improvement checked every `check` steps
"""
function StatsBase.fit!(model, X, batchsize::Int, maxsteps::Int, maxpath::Int; check = 1000, minimum_improvement = typemin(Float64), opt = ADAM(), debugfile = "", xval = X, gradmethod = :auto)
	ps = Flux.params(model)
	gradfun = getgradfun(gradmethod, model, X, batchsize, maxpath, ps)
	oldlkl = -mean(logpdf(model, xval))
	i = 0;
	train_time = 0.0
	likelihood_time = 0.0
	history = MVHistory()
	while true
		train_time += @elapsed for j in 1:check
			gs = gradfun()
			gs == nothing && @warn "nothing in gradient"
			Flux.Optimise.update!(opt, ps, gs)
		end
		i += check

		update_time = @elapsed updatelatent!(model, X, batchsize);
		likelihood_time = @elapsed newlkl = -mean(batchlogpdf(model, xval, batchsize))
		println(i,": likelihood = ", -newlkl, "  time per iteration: ", train_time / i,"s "update time",update_time,"likelihood time: ",likelihood_time)
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
	updatelatent!(model, X, batchsize);
	history
end

function samplepdf!(bestpath, model, x, repetitions::Int, pickbest = true)
	paths = [samplepath(model) for i in 1:repetitions]
	logpdfs = similar(x, size(x,2), repetitions)
	Threads.@threads for i in 1:repetitions
		logpdfs[:,i] .= pathlogpdf(model, x, paths[i])
	end


	y = mapslices(argmax, logpdfs, dims = 2)
	o = [logpdfs[i, y[i]] for i in 1:size(x,2)]
	path = [paths[y[i]] for i in 1:size(x,2)]
	bestpdf = batchpathlogpdf(model, x, bestpath)
	updatebestpath!(bestpdf, bestpath, o, path)
	(pickbest) ? bestpath :  [paths[sample(1:repetitions, Weights(softmax(logpdfs[:,i])))] for i in 1:size(x,2)]
end

function updatebestpath!(bestpdf, bestpath, o, path)
	mask = o .> bestpdf
	bestpath[mask] = path[mask]
end

function samplinggrad(model, X, bestpath, batchsize, maxpath, ps, pickbest = true)
	idxs = sample(1:size(X,2), batchsize, replace = false)
	x = X[:, idxs]
	t1 = @elapsed path = SumDenseProduct.samplepdf!(bestpath[idxs], model, x, maxpath, pickbest)
	t2 = @elapsed gr = gradient(() -> -mean(batchpathlogpdf(model, x, bestpath[idxs])), ps)
	# t2 = @elapsed gr = threadedgrad(i -> -sum(batchpathlogpdf(model, x[:,i], path[i])), ps, size(x,2))
	# println(" sampling: ",t1,"  gradient: ",t2)
	gr
end
	
function exactgrad(model, X, batchsize, ps)
	idxs = sample(1:size(X,2), batchsize, replace = false)
	x = X[:, idxs]
	# gradient(() -> -mean(logpdf(model, x)), ps)
	threadedgrad(i -> -sum(logpdf(model, x[:,i])), ps, size(x,2))
end

function exactpathgrad(model, X, batchsize, ps)
	idxs = sample(1:size(X,2), batchsize, replace = false)
	x = X[:, idxs]
	t1 = @elapsed lkl, path = mappath(model, x)
	# t2 = @elapsed gr  = gradient(() -> -mean(batchpathlogpdf(model, x, path)), ps)
	t2 = @elapsed gr  = threadedgrad(i -> -sum(batchpathlogpdf(model, x[:,i], path[i])), ps, size(x,2))
	# println(" bestpath: ",t1,"  gradient: ",t2)
	gr
end

function tunegrad(model, X, batchsize, maxpath, ps, gradmethod = [:exact, :sampling, :exactpathgrad])
	bestpath = [samplepath(model) for i in 1:size(X,2)]
	τ₁, τ₂, τ₃ = typemax(Float64), typemax(Float64), typemax(Float64)
	if :sampling ∈ gradmethod
		τ₁ = @elapsed samplinggrad(model, X, bestpath, batchsize, maxpath, ps)
		println("compilation of samplinggrad: ", τ₁)
		τ₁ = @elapsed samplinggrad(model, X, bestpath, batchsize, maxpath, ps)
		println("execution of samplinggrad: ", τ₁)
	end
	if :exact ∈ gradmethod
		τ₂ = @elapsed exactgrad(model, X, batchsize, ps)
		println("compilation of exactgrad: ", τ₂)
		τ₂ =  @elapsed exactgrad(model, X, batchsize, ps)
		println("execution of exactgrad: ", τ₂)
	end
	if :exactpath ∈ gradmethod
		τ₃ = @elapsed exactpathgrad(model, X, batchsize, ps)
		println("compilation of exactpathgrad: ", τ₃)
		τ₃ = @elapsed exactpathgrad(model, X, batchsize, ps)
		println("execution of exactpathgrad: ", τ₃)
	end
	i = argmin([τ₁, τ₂, τ₃])
	if i == 1
		println("using samplinggrad for calculation of the gradient")
		return(() -> samplinggrad(model, X, bestpath, batchsize, maxpath, ps))
	elseif i==2
		println("using exactgrad for calculation of the gradient")
		return(() -> exactgrad(model, X, batchsize, ps))
	else
		println("using exactpathgrad for calculation of the gradient")
		return(() -> exactpathgrad(model, X, batchsize, ps))

	end
end

getgradfun(gradmethod::Vector{Symbol}, model, X, batchsize, maxpath, ps) = tunegrad(model, X, batchsize, maxpath, ps, gradmethod)
function getgradfun(gradmethod::Symbol, model, X, batchsize, maxpath, ps)
	if gradmethod == :auto
		return(tunegrad(model, X, batchsize, maxpath, ps))
	elseif gradmethod == :exact
		return(() -> exactgrad(model, X, batchsize, ps))
	elseif gradmethod == :sampling
		bestpath = [samplepath(model) for i in 1:size(X,2)]
		return(() -> samplinggrad(model, X, bestpath, batchsize, maxpath, ps, false))
	elseif gradmethod == :bestsampling
		bestpath = [samplepath(model) for i in 1:size(X,2)]
		return(() -> samplinggrad(model, X, bestpath, batchsize, maxpath, ps, true))
	elseif gradmethod == :exactpath
		return(() -> exactpathgrad(model, X, batchsize, ps))
	else 
		@error "unknown gradmethod $(gradmethod)"
	end
end

function isnaninf(gs)
  for (k,v) in gs.grads
    !isa(v, AbstractArray) && continue
    (any(isinf.(v)) || any(isnan.(v))) && return(true)
  end
  return(false)
end
