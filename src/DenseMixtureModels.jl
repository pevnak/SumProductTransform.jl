module DenseMixtureModels
using Distributions, NNlib, Flux, Unitary, Zygote

const COLORS = [:blue, :red, :green, :yellow, :cyan, :magenta]

function paddedprint(io, s...; color=:default, pad=[])
    for (c, p) in pad
        printstyled(io, p, color=c)
    end
    printstyled(io, s..., color=color)
end

"""
	A fallback method
"""
function dsprint(io, s ; pad = [])
	paddedprint(io, s, pad = pad)
end


function logsumexp(x; dims = :)
	xm = maximum(x, dims = dims)
	log.(sum(exp.(x .- xm), dims = dims)) .+ xm
end

logsoftmax(x; dims = :) = x .- logsumexp(x, dims = dims)
softmax(x; dims = :) = exp.(logsoftmax(x, dims = dims))

log_normal(x) = - sum(x.^2, dims=1) / 2 .- size(x,1)*log(2π) / 2
log_normal(x,μ) = log_normal(x .- μ)
log_normal(x,μ, σ2::T) where {T<:Number} = - sum((@. ((x - μ)^2)/σ2), dims=1)/2 .- size(x,1)*log(σ2*2π)/2

include("em.jl")
include("densep.jl")
include("modelbuilders.jl")

export sharedmixture, nosharedmixture, allsharedmixture, updateprior!

end # module
