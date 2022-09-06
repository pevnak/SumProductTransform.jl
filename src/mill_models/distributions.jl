
Distributions.logpdf(d::UnivariateDistribution, x::ArrayNode) = logpdf(d, x.data)
Distributions.logpdf(d::MultivariateDistribution, x::ArrayNode) = logpdf(d, x.data)


function logfactorial(x::Integer)
    sum(log.(collect(2:x)))
end

mutable struct PoissonA{T} <: DiscreteUnivariateDistribution
    λ::Array{T, 1}
end
Flux.@functor PoissonA

PoissonA(λ::Real) = PoissonA([λ])
PoissonA(λ::Integer) = PoissonA(float(λ))


function Distributions.logpdf(m::PoissonA, x::Union{T, Array{T, 1}}) where {T <: Integer}
    f(λ, n) = n .* log.(λ) .- λ .- logfactorial.(n)
    mapreduce(λ -> f(λ, x), +, m.λ)
end
