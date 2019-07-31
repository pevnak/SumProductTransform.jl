module DenseMixtureModels
using Distributions, NNlib, Flux, Unitary

function logsumexp(x; dims = :)
	xm = maximum(x, dims = dims)
	log.(sum(exp.(x .- xm), dims = dims)) .+ xm
end

log_normal(x) = - sum(x.^2, dims=1) / 2 .- size(x,1)*log(2π) / 2
log_normal(x,μ) = log_normal(x .- μ)
log_normal(x,μ, σ2::T) where {T<:Number} = - sum((@. ((x - μ)^2)/σ2), dims=1)/2 .- size(x,1)*log(σ2*2π)/2

include("em.jl")
include("densep.jl")

end # module
