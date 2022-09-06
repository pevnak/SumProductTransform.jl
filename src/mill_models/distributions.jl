
Distributions.logpdf(d::UnivariateDistribution, x::ArrayNode) = logpdf(d, x.data)
Distributions.logpdf(d::MultivariateDistribution, x::ArrayNode) = logpdf(d, x.data)


function logfactorial(x::Integer)
    sum(log.(2:x))
end


mutable struct PoissonA{T} <: DiscreteUnivariateDistribution
    λ::Array{T, 0}
end

PoissonA(λ::Real) = PoissonA(fill(λ))
PoissonA(λ::Integer) = PoissonA(float(λ))


function Distributions.logpdf(m::PoissonA, x::Union{T, Array{T, 1}}) where {T <: Integer}
    x .* log.(m.λ) .- m.λ .- logfactorial.(x)
end
