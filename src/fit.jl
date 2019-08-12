using IterTools, BSON
"""
	fit!(model, X, batchsize::Int, maxsteps::Int; minimum_improvement = 1e-4, opt = ADAM(), check = 1000)

	fits the model using stochastic gradient descend with on data `X` using stochastic
	gradient descend with `batchsize` executed for `maxsteps` with improvement checked every `check` steps
"""
function StatsBase.fit!(model, X, batchsize::Int, maxsteps::Int; check = 1000, minimum_improvement = 1e-4, opt = ADAM(), debugfile = "")
	dataiterator = repeatedly(() -> (X[:, sample(1:size(X,2), batchsize, replace = false)],), check)
	ps = Flux.params(model)
	cb = Flux.throttle(() -> (@show mean(logpdf(model, X))),10)
	oldlkl = -mean(logpdf(model, X))
	i = 0;
	while true
		if isempty(debugfile)
			Flux.train!(x -> -mean(logpdf(model, x)), ps, dataiterator, opt)
		else
			debugtrain!((model, x) -> -mean(logpdf(model, x)), model, ps, dataiterator, opt)
		end
		i += check
		newlkl = -mean(logpdf(model, X))
		println(i,": likelihood = ", -newlkl)
		if oldlkl - newlkl < minimum_improvement 
			@info "breaking after $(i) steps because of minimum improvement not met"
			break;
		end
		if i >= maxsteps
			@info "breaking after $(i) steps because of maximum number of steps exceeded"
			break;
		end
		oldlkl = newlkl
	end
	updatelatent!(model, X);
	model
end

function debugtrain!(loss, model, ps, data, opt; cb = () -> (), debugfile = "")
  ps = Params(ps)
  cb = Flux.Optimise.runall(cb)
for d in data
    try
      gs = gradient(() -> loss(model, d...), ps)
      if !isempty(debugfile) && isnaninf(gs)
      	BSON.@save debugfile model data loss
      	@error "Nan or Inf in the gradient, debug info stored in $debugfile"
      end                  
      Flux.Optimise.update!(opt, ps, gs)
      if cb() == :stop
        depwarn("Use of `:stop` is deprecated; use `Flux.stop()` instead", :stop)
        break
      end
    catch ex
      if ex isa Flux.Optimise.StopException
        break
      else
        rethrow(ex)
      end
    end
  end
end

function isnaninf(gs)
  for (k,v) in gs.grads
    !isa(v, AbstractArray) && continue
    (any(isinf.(v)) || any(isnan.(v))) && return(true)
  end
  return(false)
end
