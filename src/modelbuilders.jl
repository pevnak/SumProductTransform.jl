function createmixture(d::Int, n::Int, σ = identity, p = d -> MultivariateNormal(d,1))
	SumNode([DenseNode(Unitary.SVDDense(d, σ), p()) for i in 1:n], fill(1/n, n))
end

addnoise(noisedim, pnoise, p) = noisedim == 0 ? p : ProductNode((pnoise(noisedim), p))

"""
	nosharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MultivariateNormal(d,1))

	There is not sharing here, as every children uses its own distributions
"""
function nosharedmixture(d::Int, ns::Vector{Int}, σs::Vector, p = d -> MultivariateNormal(d,1))
	n, σ = ns[1], σs[1]
	length(ns) == 1 && return(SumNode([DenseNode(Unitary.SVDDense(d, σ), p(d)) for i in 1:n], fill(1/n, n)))
	return(SumNode([DenseNode(Unitary.SVDDense(d, σ), nosharedmixture(d, ns[2:end], σs[2:end], p)) for i in 1:n], fill(1/n, n)))
end
nosharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MultivariateNormal(d,1)) = nosharedmixture(d, fill(n,l), fill(σ,l), p)

function nosharedmixture(d::Int, ns::Vector{Int}, σs::Vector, noise::Vector, p = d -> MultivariateNormal(d,1))
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
	SumNode(components, fill(1/n, n))
end

"""
	allsharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MultivariateNormal(d,1))

	There is not sharing here, as every children uses its own distributions
"""
function allsharedmixture(d::Int, ns::Vector{Int}, σs::Vector, p = d -> MultivariateNormal(d,1))
	@assert length(ns) == length(σs)
	n, σ = ns[end], σs[end]
	m = SumNode([DenseNode(Unitary.SVDDense(d, σ), p(d)) for i in 1:n], fill(1/n, n))
	for i in length(ns)-1:-1:1
		n, σ = ns[i], σs[i]
		m = SumNode([DenseNode(Unitary.SVDDense(d, σ), m) for i in 1:n], fill(1/n, n))
	end
	m
end
allsharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MultivariateNormal(d,1)) = allsharedmixture(d, fill(n,l), fill(σ,l), p)

function allsharedmixture(d::Int, ns::Vector{Int}, σs::Vector, noise::Vector, p = d -> MultivariateNormal(d,1))
	@assert length(ns) == length(σs)
	n, σ, noisedim = ns[end], σs[end], noise[end]
	noisedim > 0 && @warn "We ignore the noise in last layer (they are independent anyway)"
	truedim = d - sum(noise)
	m = SumNode([DenseNode(Unitary.SVDDense(truedim, σ), p(truedim)) for i in 1:n], fill(1/n, n))
	for i in length(ns)-1:-1:1
		n, σ, noisedim = ns[i], σs[i], noise[i]
		truedim = d - sum(noise[1:i-1])
		m = SumNode([DenseNode(Unitary.SVDDense(truedim, σ), addnoise(noisedim, p, m)) for i in 1:n], fill(1/n, n))
	end
	m
end

"""
	densesharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MultivariateNormal(d,1))

	the models share the dense non-linear layers, but they do not share components weights (priors)
"""
function densesharedmixture(d::Int, ns::Vector{Int}, σs::Vector, p = d -> MultivariateNormal(d,1))
	@assert length(ns) == length(σs)
	n, σ, nprevious = ns[end], σs[end], ns[end]
	non_linear_part = [DenseNode(Unitary.SVDDense(d, σ), p) for i in 1:n];
	for i in length(ns)-1:-1:1
		n, σ = ns[i], σs[i]
		non_linear_part = [DenseNode(Unitary.SVDDense(d, σ), SumNode(non_linear_part, fill(1/nprevious, nprevious))) for i in 1:n];
		nprevious = n
	end
	m = SumNode(non_linear_part, fill(1/n, n))
	m
end

function densesharedmixture(d::Int, ns::Vector{Int}, σs::Vector, noise::Vector, p = d -> MultivariateNormal(d,1))
	@assert length(ns) == length(σs)
	n, σ, nprevious, noisedim = ns[end], σs[end], ns[end], noise[end]
	noisedim > 0 && @warn "We ignore the noise in last layer (they are independent anyway)"
	truedim = d - sum(noise)
	non_linear_part = [DenseNode(Unitary.SVDDense(truedim, σ), p(truedim)) for i in 1:n];
	for i in length(ns)-1:-1:1
		n, σ, noisedim = ns[i], σs[i], noise[i]
		truedim = d - sum(noise[1:i-1])
		non_linear_part = [DenseNode(Unitary.SVDDense(truedim, σ), addnoise(noisedim, p, SumNode(non_linear_part, fill(1/nprevious, nprevious)))) for i in 1:n];
		nprevious = n
	end
	m = SumNode(non_linear_part, fill(1/n, n))
	m
end

densesharedmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MultivariateNormal(d,1)) = densesharedmixture(d, fill(n,l), fill(σ,l), p)

function buildmixture(d::Int, n::Int, l::Int, σ = identity, p = d -> MultivariateNormal(d,1); sharing = :all)
	if sharing == :all 
		allsharedmixture(d, n, l, σ, p)
	elseif sharing == :dense
		densesharedmixture(d, n, l, σ, p)
	elseif sharing == :none 
		nosharedmixture(d, n, l, σ, p)
	else 
		@error "unknown sharing $(sharing)"
	end
end

function buildmixture(d::Int, n::Vector, l::Vector, p = d -> MultivariateNormal(d,1); sharing = :all)
	if sharing == :all 
		allsharedmixture(d, n, l, p)
	elseif sharing == :dense
		densesharedmixture(d, n, l, p)
	elseif sharing == :none 
		nosharedmixture(d, n, l, p)
	else 
		@error "unknown sharing $(sharing)"
	end
end