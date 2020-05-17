using Unitary
addnoise(noisedim, pnoise, p) = noisedim == 0 ? p : ProductNode((pnoise(noisedim), p))

"""
	nosharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MvNormal(d,1f0))

	There is not sharing here, as every children uses its own distributions
"""
function nosharedmixture(d::Int, ns::Vector{Int}, σs::Vector, noise::Vector, p = d -> MvNormal(d,1f0), unitary = :householder)
	@assert length(ns) == length(σs) == length(noise) 
	@assert sum(noise) <= d
	n, σ, noisedim = ns[1], σs[1], noise[1]
	components = if length(ns) == 1
		noisedim > 0 && @warn "We ignore the noise in last layer (they are independent anyway)"
		[TransformNode(Chain(Unitary.Transform(d, σ, unitary), Unitary.Transform(d, identity, unitary))s, p(d)) for i in 1:n]
	else
		ns, σs, noise = ns[2:end], σs[2:end], noise[2:end]
		[TransformNode(Chain(Unitary.Transform(d, σ, unitary), Unitary.Transform(d, identity, unitary))s, 
			addnoise(noisedim, p, nosharedmixture(d - noisedim, ns, σs, noise, p, unitary)))
			for i in 1:n]
	end
	SumNode(components)
end

"""
	allsharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MvNormal(d,1f0))

	There is not sharing here, as every children uses its own distributions
"""
function allsharedmixture(d::Int, ns::Vector{Int}, σs::Vector, noise::Vector, p = d -> MvNormal(d,1f0), unitary = :householder)
	@assert length(ns) == length(σs) == length(noise)
	n, σ, noisedim = ns[end], σs[end], noise[end]
	noisedim > 0 && @warn "We ignore the noise in last layer (they are independent anyway)"
	noise[end] = 0
	truedim = d - sum(noise)
	m = SumNode([TransformNode(Chain(Unitary.Transform(truedim, σ, unitary), Unitary.Transform(truedim, identity, unitary)), p(truedim)) for i in 1:n])
	for i in length(ns)-1:-1:1
		n, σ, noisedim = ns[i], σs[i], noise[i]
		truedim = d - sum(noise[1:i-1])
		m = SumNode([TransformNode(Chain(Unitary.Transform(truedim, σ, unitary), Unitary.Transform(truedim, identity, unitary)), addnoise(noisedim, p, m)) for i in 1:n])
	end
	m
end
"""
	transformsharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MvNormal(d,1f0))

	the models share the transform non-linear layers, but they do not share components weights (priors)
"""
function transformsharedmixture(d::Int, ns::Vector{Int}, σs::Vector, noise::Vector, p = d -> MvNormal(d,1f0), unitary = :householder)
	@assert length(ns) == length(σs) == length(noise)
	n, σ, noisedim = ns[end], σs[end], noise[end]
	noisedim > 0 && @warn "We ignore the noise in last layer (they are independent anyway)"
	noise[end] = 0
	truedim = d - sum(noise)
	non_linear_part = [TransformNode(Chain(Unitary.Transform(truedim, σ, unitary), Unitary.Transform(truedim, identity, unitary)), p(truedim)) for i in 1:n];
	for i in length(ns)-1:-1:1
		n, σ, noisedim = ns[i], σs[i], noise[i]
		truedim = d - sum(noise[1:i-1])
		non_linear_part = [TransformNode(Chain(Unitary.Transform(truedim, σ, unitary), Unitary.Transform(truedim, identity, unitary)), addnoise(noisedim, p, SumNode(non_linear_part))) for i in 1:n];
	end
	m = SumNode(non_linear_part)
	m
end

function buildmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MvNormal(d,1f0); sharing = :all, firsttransform = false, unitary = :householder)
	ns = fill(n, l)
	σs = fill(σ, l)
	noise = fill(0, l)
	model = if sharing == :all 
		allsharedmixture(d, ns, σs, noise, p, unitary)
	elseif sharing == :transform
		transformsharedmixture(d, ns, σs, noise, p, unitary)
	elseif sharing == :none 
		nosharedmixture(d, ns, σs, noise, p, unitary)
	else 
		@error "unknown sharing $(sharing)"
	end
    model = firsttransform ? TransformNode(Chain(Unitary.Transform(d, σ, unitary), Unitary.Transform(d, identity, unitary))s, model) : model
end

function buildmixture(d::Int, n::Vector, l::Vector, noise::Vector = fill(0, length(n)),  p = d -> MvNormal(d,1f0); sharing = :all, firsttransform = false, unitary = :householder)
	model = if sharing == :all 
		allsharedmixture(d, n, l, noise, p, unitary)
	elseif sharing == :transform
		transformsharedmixture(d, n, l, noise, p, unitary)
	elseif sharing == :none 
		nosharedmixture(d, n, l, noise, p, unitary)
	else 
		@error "unknown sharing $(sharing)"
	end
	model = firsttransform ? TransformNode(Chain(Unitary.Transform(d, σ, unitary), Unitary.Transform(d, identity, unitary))s, model) : model
end