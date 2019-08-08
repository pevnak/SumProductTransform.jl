

function createmixture(d::Int, n::Int, σ = identity, p = () -> MultivariateNormal(2,1))
	SumNode([DenseNode(Unitary.SVDDense(d, σ), p()) for i in 1:n], fill(1/n, n))
end

"""
	nosharedmixture(d::Int, n::Int, l::Int, σ = identity, p = MultivariateNormal(d,1))

	There is not sharing here, as every children uses its own distributions
"""
function nosharedmixture(d::Int, ns::Vector{Int}, σs::Vector, p = MultivariateNormal(d,1))
	n, σ = ns[1], σs[1]
	length(ns) == 1 && return(SumNode([DenseNode(Unitary.SVDDense(d, σ), p) for i in 1:n], fill(1/n, n)))
	return(SumNode([DenseNode(Unitary.SVDDense(d, σ), nosharedmixture(d, ns[2:end], σs[2:end], p)) for i in 1:n], fill(1/n, n)))
end
nosharedmixture(d::Int, n::Int, l::Int, σ = identity, p = MultivariateNormal(d,1)) = nosharedmixture(d, fill(n,l), fill(σ,l), p)

"""
	allsharedmixture(d::Int, n::Int, l::Int, σ = identity, p = MultivariateNormal(d,1))

	There is not sharing here, as every children uses its own distributions
"""
function allsharedmixture(d::Int, ns::Vector{Int}, σs::Vector, p = MultivariateNormal(d,1))
	@assert length(ns) == length(σs)
	n, σ = ns[end], σs[end]
	m = SumNode([DenseNode(Unitary.SVDDense(d, σ), p) for i in 1:n], fill(1/n, n))
	for i in length(ns)-1:-1:1
		n, σ = ns[i], σs[i]
		m = SumNode([DenseNode(Unitary.SVDDense(d, σ), m) for i in 1:n], fill(1/n, n))
	end
	m
end
allsharedmixture(d::Int, n::Int, l::Int, σ = identity, p = MultivariateNormal(d,1)) = allsharedmixture(d, fill(n,l), fill(σ,l), p)

"""
	densesharedmixture(d::Int, n::Int, l::Int, σ = identity, p = MultivariateNormal(d,1))

	the models share the dense non-linear layers, but they do not share components weights (priors)
"""
function densesharedmixture(d::Int, ns::Vector{Int}, σs::Vector, p = MultivariateNormal(d,1))
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

densesharedmixture(d::Int, n::Int, l::Int, σ = identity, p = MultivariateNormal(d,1)) = densesharedmixture(d, fill(n,l), fill(σ,l), p)

function buildmixture(d::Int, n::Int, l::Int, σ = identity, p = MultivariateNormal(d,1); sharing = :all)
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

function buildmixture(d::Int, n::Vector, l::Vector, p = MultivariateNormal(d,1); sharing = :all)
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