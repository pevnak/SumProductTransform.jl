using Unitary
addnoise(noisedim, pnoise, p) = noisedim == 0 ? p : ProductNode((pnoise(noisedim), p))

"""
	nosharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MvNormal(d,1f0))

	There is not sharing here, as every children uses its own distributions
"""
function nosharedmixture(d::Int, ns::Vector{Int}, σs::Vector, noise::Vector, p = d -> MvNormal(d,1f0))
	@assert length(ns) == length(σs) == length(noise) 
	@assert sum(noise) <= d
	n, σ, noisedim = ns[1], σs[1], noise[1]
	components = if length(ns) == 1
		noisedim > 0 && @warn "We ignore the noise in last layer (they are independent anyway)"
		[DenseNode(Unitary.SVDDense(d, σ), p(d)) for i in 1:n]
	else
		ns, σs, noise = ns[2:end], σs[2:end], noise[2:end]
		[DenseNode(Unitary.SVDDense(d, σ), 
			addnoise(noisedim, p, nosharedmixture(d - noisedim, ns, σs, noise, p)))
			for i in 1:n]
	end
	SumNode(components)
end
nosharedmixture(d, ns::Vector, σs::Vector) = nosharedmixture(d, ns, σs, fill(0, length(ns)), d -> MvNormal(d,1f0))
nosharedmixture(d::Int, n::Int, l::Int, σ = identity) = nosharedmixture(d, fill(n,l), fill(σ,l))


"""
	allsharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MvNormal(d,1f0))

	There is not sharing here, as every children uses its own distributions
"""
function allsharedmixture(d::Int, ns::Vector{Int}, σs::Vector, noise::Vector, p = d -> MvNormal(d,1f0))
	@assert length(ns) == length(σs) == length(noise)
	n, σ, noisedim = ns[end], σs[end], noise[end]
	noisedim > 0 && @warn "We ignore the noise in last layer (they are independent anyway)"
	noise[end] = 0
	truedim = d - sum(noise)
	m = SumNode([DenseNode(Unitary.SVDDense(truedim, σ), p(truedim)) for i in 1:n])
	for i in length(ns)-1:-1:1
		n, σ, noisedim = ns[i], σs[i], noise[i]
		truedim = d - sum(noise[1:i-1])
		m = SumNode([DenseNode(Unitary.SVDDense(truedim, σ), addnoise(noisedim, p, m)) for i in 1:n])
	end
	m
end
allsharedmixture(d, ns::Vector, σs::Vector) = allsharedmixture(d, ns, σs, fill(0, length(ns)), d -> MvNormal(d,1f0))
allsharedmixture(d::Int, n::Int, l::Int, σ = identity) = allsharedmixture(d, fill(n,l), fill(σ,l))

"""
	densesharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MvNormal(d,1f0))

	the models share the dense non-linear layers, but they do not share components weights (priors)
"""
function densesharedmixture(d::Int, ns::Vector{Int}, σs::Vector, noise::Vector, p = d -> MvNormal(d,1f0))
	@assert length(ns) == length(σs) == length(noise)
	n, σ, noisedim = ns[end], σs[end], noise[end]
	noisedim > 0 && @warn "We ignore the noise in last layer (they are independent anyway)"
	noise[end] = 0
	truedim = d - sum(noise)
	non_linear_part = [DenseNode(Unitary.SVDDense(truedim, σ), p(truedim)) for i in 1:n];
	for i in length(ns)-1:-1:1
		n, σ, noisedim = ns[i], σs[i], noise[i]
		truedim = d - sum(noise[1:i-1])
		non_linear_part = [DenseNode(Unitary.SVDDense(truedim, σ), addnoise(noisedim, p, SumNode(non_linear_part))) for i in 1:n];
	end
	m = SumNode(non_linear_part)
	m
end
densesharedmixture(d, ns::Vector, σs::Vector) = densesharedmixture(d, ns, σs, fill(0, length(ns)), d -> MvNormal(d,1f0))
densesharedmixture(d::Int, n::Int, l::Int, σ = identity) = densesharedmixture(d, fill(n,l), fill(σ,l))

function buildmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MvNormal(d,1f0); sharing = :all)
	ns = fill(n, l)
	σs = fill(σ, l)
	noise = fill(0, l)
	if sharing == :all 
		allsharedmixture(d, ns, σs, noise, p)
	elseif sharing == :dense
		densesharedmixture(d, ns, σs, noise, p)
	elseif sharing == :none 
		nosharedmixture(d, ns, σs, noise, p)
	else 
		@error "unknown sharing $(sharing)"
	end
end

function buildmixture(d::Int, n::Vector, l::Vector, noise::Vector = fill(0, length(n)),  p = d -> MvNormal(d,1f0); sharing = :all)
	if sharing == :all 
		allsharedmixture(d, n, l, noise, p)
	elseif sharing == :dense
		densesharedmixture(d, n, l, noise, p)
	elseif sharing == :none 
		nosharedmixture(d, n, l, noise, p)
	else 
		@error "unknown sharing $(sharing)"
	end
end