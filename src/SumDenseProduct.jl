module SumDenseProduct
using Distributions, NNlib, Flux, Unitary, Zygote, StatsBase, FillArrays

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

log_normal(x) = - sum(x.^2, dims=1) / 2 .- size(x,1)*log(Float32(2π)) / 2
log_normal(x,μ) = log_normal(x .- μ)
log_normal(x,μ, σ2::T) where {T<:Number} = - sum((@. ((x - μ)^2)/σ2), dims=1)/2 .- size(x,1)*log(σ2*2π)/2

#Let's do a little bit of function stealing
Distributions.logpdf(p::MvNormal, x::AbstractMatrix) = log_normal(x)[:]

batchlogpdf(p, x, bs::Int) = reduce(vcat, map(i -> logpdf(p, x[:,i]), Iterators.partition(1:size(x,2), bs)))


"""
    treecount(m)

    Number of possible tree of a model. For distributions it defaults to one.
"""
treecount(m) = 1

"""
    treelogpdf(p, x, tree)

    logpdf of samples `x` calculated along the `tree` determining components in sumnodes (at the moment)
    For distributions outside the SumDenseProduct it falls back to logpdf(p, x).
"""
treelogpdf(m, x, tree) = logpdf(m, x)

"""
    tree = sampletree(m)

    sample a tree trough the model, which can be used by treelogpdf to calculate the
    pdf along this tree.
"""
sampletree(m) = tuple()

_maptree(m, x)= (logpdf(m,x), fill(tuple(), size(x, 2)))

batchtreelogpdf(m, x, tree) = map(i -> treelogpdf(m, x[:,i:i], tree[i])[1], 1:length(tree))


_priors(m) = nothing
function priors(m)
  ps = Flux.Params()
  Flux.prefor(p -> begin
  	pr = _priors(p)
  	pr != nothing && !any(p′ -> p′ === pr, ps) && push!(ps, pr)
  end, m)
  return ps
end


include("threadedgrads.jl")
include("sumnode.jl")
include("transformnode.jl")
include("productnode.jl")
include("modelbuilders.jl")
include("fit.jl")
include("updatelatent.jl")
#include("smartinit.jl")


export SumNode, TransformNode, ProductNode
export transformsharedmixture, nosharedmixture, allsharedmixture, priors, updatelatent!, buildmixture, treecount, batchlogpdf, initpp!

end # module
