using IterTools, BSON, ValueHistories, MLDataPattern
using TimerOutputs
const to = TimerOutput()
"""
	fit!(model, X, batchsize::Int, maxsteps::Int, maxtree::Int; check = 1000, minimum_improvement = typemin(Float64), opt = ADAM(), debugfile = "", xval = X)

	fits the model using stochastic gradient descend with on data `X` using stochastic
	gradient descend with `batchsize` executed for `maxsteps` with improvement checked every `check` steps
"""
function StatsBase.fit!(model, X, batchsize::Int, maxsteps::Int, maxtree::Int; check = 1000, minimum_improvement = typemin(Float64), opt = ADAM(), debugfile = "", xval = X, gradmethod = :exact)
	fit!(model, X, batchsize::Int, maxsteps::Int; maxtree, check, minimum_improvement, opt, debugfile, xval, gradmethod)
end

function StatsBase.fit!(model, X, batchsize::Int, maxsteps::Int; maxtree = 100, check = 1000, minimum_improvement = typemin(Float64), opt = ADAM(), debugfile = "", xval = X, gradmethod = :exact, ps = Flux.params(model))
	gradfun = getgradfun(gradmethod, model, X, batchsize, maxtree, ps)
	# oldlkl = -mean(logpdf(model, xval))
	oldlkl = 0
	i = 0;
	train_time = 0.0
	likelihood_time = 0.0
	history = MVHistory()
	while true
		check = min(check, maxsteps - i)
		train_time += @elapsed for j in 1:check
			gs = gradfun()
			Flux.Optimise.update!(opt, ps, gs)
		end
		i += check

		# update_time = @elapsed updatelatent!(model, X, batchsize);
		update_time = 0
		likelihood_time = @elapsed newlkl = - mean(batchlogpdf(model, xval, batchsize))
		# newlkl = 0
		likelihood_time = 0
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

function samplepdf!(besttree, model, x, repetitions::Int, pickbest::Bool = true)
	trees = [sampletree(model) for i in 1:repetitions]
	logpdfs = similar(x, size(x,2), repetitions)
	@timeit to "treelogpdf" Threads.@threads for i in 1:repetitions
		logpdfs[:,i] .= treelogpdf(model, x, trees[i])
	end

	if pickbest 		
		y = mapslices(argmax, logpdfs, dims = 2)
		o = [logpdfs[i, y[i]] for i in 1:size(x,2)]
		tree = [trees[y[i]] for i in 1:size(x,2)]
		@timeit to "batchtreelogpdf" bestpdf = batchtreelogpdf(model, x, besttree)
		updatebesttree!(bestpdf, besttree, o, tree)
		return(besttree)
	else 
		return([trees[sample(1:repetitions, Weights(softmax(logpdfs[i,:])))] for i in 1:size(x,2)])
	end
end

function samplepdf!(model, x, repetitions::Int, pickbest::Bool = true)
	trees = [sampletree(model) for i in 1:repetitions]
	logpdfs = similar(x, size(x,2), repetitions)
	@timeit to "treelogpdf" Threads.@threads for i in 1:repetitions
		logpdfs[:,i] .= treelogpdf(model, x, trees[i])
	end

	if pickbest 		
		y = mapslices(argmax, logpdfs, dims = 2)
		o = [logpdfs[i, y[i]] for i in 1:size(x,2)]
		tree = [trees[y[i]] for i in 1:size(x,2)]
		return(tree)
	else 
		return([trees[sample(1:repetitions, Weights(softmax(logpdfs[i,:])))] for i in 1:size(x,2)])
	end
end

function updatebesttree!(bestpdf, besttree, o, tree)
	mask = o .> bestpdf
	besttree[mask] = tree[mask]
end

function samplinggrad(model, X, besttree, batchsize, maxtree, ps, pickbest::Bool = true)
	@timeit to "samplinggrad" begin
		idxs = sample(1:size(X,2), batchsize, replace = false)
		x = X[:, idxs]
		@timeit to "samplepdf" tree = SumProductTransform.samplepdf!(view(besttree,idxs), model, x, maxtree, pickbest)
		bp = besttree[idxs]
		@timeit to "gradient" threadedgrad(i -> -sum(batchtreelogpdf(model, x[:,i], bp[i])), ps, size(x,2))
	end
end
	
function samplinggrad(model, X, batchsize, maxtree, ps, pickbest::Bool = true)
	@timeit to "samplinggrad" begin
		idxs = sample(1:size(X,2), batchsize, replace = false)
		x = X[:, idxs]
		@timeit to "samplepdf" tree = SumProductTransform.samplepdf!(model, x, maxtree, pickbest)
		@timeit to "gradient" threadedgrad(i -> -sum(batchtreelogpdf(model, x[:,i], tree[i])), ps, size(x,2))
	end
end
	
function exactgrad(model, X, batchsize, ps)
	@timeit to "exactgrad" begin
		idxs = sample(1:size(X,2), batchsize, replace = false)
		x = X[:, idxs]
		# @timeit to "gradient" threadedgrad(i -> -sum(logpdf(model, x[:,i])), ps, size(x,2))
		gs = gradient(() -> -mean(logpdf(model, x)), ps)
		# if any([any(isnan.(gs[p])) for p in ps])
		# 	!isfile("/tmp/sumdense.jls") && serialize("/tmp/sumdense.jls",(model, x))
		# 	@error "nan in gradient"
		# end
		gs
	end
end

function exacttreegrad(model, X, batchsize, ps)
	@timeit to "exacttreegrad" begin
		idxs = sample(1:size(X,2), batchsize, replace = false)
		x = X[:, idxs]
		@timeit to "maptree" lkl, tree = maptree(model, x)
		@timeit to "gradient" threadedgrad(i -> -sum(batchtreelogpdf(model, x[:,i], tree[i])), ps, size(x,2))
	end
end

function tunegrad(model, X, batchsize, maxtree, ps, gradmethod = [:exact, :sampling, :exacttreegrad])
	besttree = [sampletree(model) for i in 1:size(X,2)]
	τ₁, τ₂, τ₃ = typemax(Float64), typemax(Float64), typemax(Float64)
	@show gradmethod
	if :samplingbest ∈ gradmethod
		τ₁ = @elapsed samplinggrad(model, X, besttree, batchsize, maxtree, ps)
		println("compilation of samplinggrad: ", τ₁)
		τ₁ = @elapsed samplinggrad(model, X, besttree, batchsize, maxtree, ps)
		println("execution of samplinggrad: ", τ₁)
	end
	if :sampling ∈ gradmethod
		τ₁ = @elapsed samplinggrad(model, X, batchsize, maxtree, ps)
		println("compilation of samplinggrad: ", τ₁)
		τ₁ = @elapsed samplinggrad(model, X, batchsize, maxtree, ps)
		println("execution of samplinggrad: ", τ₁)
	end
	if :exact ∈ gradmethod
		τ₂ = @elapsed exactgrad(model, X, batchsize, ps)
		println("compilation of exactgrad: ", τ₂)
		τ₂ =  @elapsed exactgrad(model, X, batchsize, ps)
		println("execution of exactgrad: ", τ₂)
	end
	if :exacttree ∈ gradmethod
		τ₃ = @elapsed exacttreegrad(model, X, batchsize, ps)
		println("compilation of exacttreegrad: ", τ₃)
		τ₃ = @elapsed exacttreegrad(model, X, batchsize, ps)
		println("execution of exacttreegrad: ", τ₃)
	end
	i = argmin([τ₁, τ₂, τ₃])
	if i == 1
		println("using samplinggrad for calculation of the gradient")
		if gradmethod == :samplingbest
			return(() -> samplinggrad(model, X, besttree, batchsize, maxtree, ps))
		else
			return(() -> samplinggrad(model, X, batchsize, maxtree, ps))
		end
	elseif i==2
		println("using exactgrad for calculation of the gradient")
		return(() -> exactgrad(model, X, batchsize, ps))
	else
		println("using exacttreegrad for calculation of the gradient")
		return(() -> exacttreegrad(model, X, batchsize, ps))

	end
end

getgradfun(gradmethod::Vector{Symbol}, model, X, batchsize, maxtree, ps) = tunegrad(model, X, batchsize, maxtree, ps, gradmethod)
function getgradfun(gradmethod::Symbol, model, X, batchsize, maxtree, ps)
	batchsize = min(size(X,2), batchsize)
	if gradmethod == :auto
		return(tunegrad(model, X, batchsize, maxtree, ps))
	elseif gradmethod == :exact
		return(() -> exactgrad(model, X, batchsize, ps))
	elseif gradmethod == :sampling
		return(() -> samplinggrad(model, X, batchsize, maxtree, ps, true))
	elseif gradmethod == :samplingbest
		besttree = [sampletree(model) for i in 1:size(X,2)]
		return(() -> samplinggrad(model, X, besttree, batchsize, maxtree, ps, true))
	elseif gradmethod == :exacttree
		return(() -> exacttreegrad(model, X, batchsize, ps))
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
