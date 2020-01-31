using Clustering, IterTools, Serialization

function removenothing(x)
	mask = isnothing.(x)
	!any(mask) && return(x)
	mask = sum(mask, dims = 1) .== 0
	xx = x[:, mask[:]]
	Float64.(xx)
end

function init_dense_only!(m::DenseNode, X)
	xx = removenothing(X)
	size(X, 2) == 0 && return(X)
	_initpp!(m, xx)
end

init_dense_only!(m, X) = X

function _initpp!(m::DenseNode, X)
	d, l = size(X)
	l == 0 && return(X)
	p = MvNormal(d, 1.0)
	ps = Flux.params(m.m)
	data = IterTools.repeatedly(() -> (X[:,sample(1:l, min(l,100))],), 1000)
	Flux.train!(x -> - mean(logpdf(p, m.m(x))), ps, data, ADAM())
	m.m(X)
end

function initpp!(m::DenseNode, X, shared)
	initpp!(m.p, X, shared)
end


function initpp!(m::ProductNode, X, shared = :none)
	for (i, interval) in enumerate(m.dimensions)
		@show (typeof(X), size(X), interval)
		initpp!(m.components[i], X[interval, :], shared)
	end
end

function initpp!(m::SumNode, X, shared)
	# try 
		size(X, 2) == 0 && return(m)
		R = kmeans(X, length(m.components); maxiter=20, display=:none)
		if shared == :none
			for i in 1:length(m.components)
				xx = init_dense_only!(m.components[i], X[:, assignments(R) .== i])
				initpp!(m.components[i].p, xx, shared)
			end
		else
			xxs = [init_dense_only!(m.components[i], X[:, assignments(R) .== i]) for i in 1:length(m.components)]
			xx = reduce(hcat, xxs)
			initpp!(m.components[1], xx, shared)
		end
	# catch
	# 	@info "smartinit has failed somewhere"
	# end
end


function initpp2!(m, X, nclusters)
	R = kmeans(X, nclusters; maxiter=20, display=:none)
	for i in 1:nclusters
		xx = X[:, assignments(R) .== i]
		initpath!(m, xx, samplepath(m))
	end
end

initpath!(m, x, path; bs = 100, nsteps = 1000, verbose::Bool = false) = initpath!(m, x, path, Flux.params(m); bs = 100, nsteps = 1000, verbose = false)

function initpath!(m, x, path, ps; bs = 100, nsteps = 1000, verbose::Bool = false)
	verbose && @show pathlogpdf(m, x, path)
	d, l = size(x)
	l == 0 && return(x)
	data = (l < bs) ? repeatedly(() -> (x,), nsteps) : repeatedly(() -> (x[:,sample(1:l, bs)],), nsteps)
	Flux.train!(xx -> - mean(pathlogpdf(m, xx, path)), ps, data, ADAM())
	verbose && @show pathlogpdf(m, x, path)
end


function _initpath!(m, x, path, ps; bs = 100, nsteps = 1000, verbose::Bool = false)
	verbose && @show pathlogpdf(m, x, path)
	for i in 1:nsteps
		gs = gradient(() -> mean(pathlogpdf(m, xx, path)), ps)
		any([any(isnan.(gs[p])) for p in ps]) && serialize("/tmp/debug/jls", (m, xx, path, ps))
		any([any(isinf.(gs[p])) for p in ps]) && serialize("/tmp/debug/jls", (m, xx, path, ps))
		Flux.Optimise.update!(opt, ps, gs)
	end
	verbose && @show pathlogpdf(m, x, path)
end




initpp!(::MvNormal, X, shared) = nothing