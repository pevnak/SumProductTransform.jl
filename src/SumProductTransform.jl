module SumProductTransform
using Distributions
using NNlib
using Flux
using Unitary
using Zygote
using StatsBase
using FillArrays
using Bijectors
using HierarchicalUtils
using HierarchicalUtils: NodeType, InnerNode, LeafNode, printchildren

function logsumexp(x; dims = :)
	xm = maximum(x, dims = dims)
	log.(sum(exp.(x .- xm), dims = dims)) .+ xm
end

logsoftmax(x; dims = :) = x .- logsumexp(x, dims = dims)
softmax(x; dims = :) = exp.(logsoftmax(x, dims = dims))
samplebatch(x, bs) = x[:, sample(1:size(x, 2), min(size(x, 2), bs), replace = false)]
sampletrees(m, bs) = [sampletree(m) for _ in 1:bs]

include("layers/layers.jl")
include("priors.jl")
include("distributions.jl")
include("threadedgrads.jl")
include("sumnode.jl")
include("transformationnode.jl")
include("productnode.jl")
include("modelbuilders.jl")
include("fit.jl")
include("smartinit.jl")
include("fitting/em.jl")
include("fitting/mhsaem.jl")


Distributions.logpdf(m::T, x::AbstractVector) where {T<:Union{SumNode, TransformationNode, ProductNode}} = logpdf(m, reshape(x, :, 1))[1]
Base.show(io::IO, ::MIME"text/plain", n::Union{SumNode, TransformationNode, ProductNode}) = HierarchicalUtils.printtree(io, n)


export SumNode, TransformationNode, ProductNode
export densesharedmixture, nosharedmixture, allsharedmixture, priors, updatelatent!, buildmixture, pathcount, batchlogpdf
export em!, fit!, mhsaem!
export SVDNode, ScaleShift

end # module
