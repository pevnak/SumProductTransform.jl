module SumProductTransform
using Distributions, NNlib, Flux, Unitary, Zygote, StatsBase, FillArrays
using HierarchicalUtils
using HierarchicalUtils: NodeType, InnerNode, LeafNode, printchildren

function logsumexp(x; dims = :)
	xm = maximum(x, dims = dims)
	log.(sum(exp.(x .- xm), dims = dims)) .+ xm
end

logsoftmax(x; dims = :) = x .- logsumexp(x, dims = dims)
softmax(x; dims = :) = exp.(logsoftmax(x, dims = dims))


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

Distributions.logpdf(m::T, x::AbstractVector) where {T<:Union{SumNode, TransformationNode, ProductNode}} = logpdf(m, reshape(x, :, 1))[1]
Base.show(io::IO, ::MIME"text/plain", n::Union{SumNode, TransformationNode, ProductNode}) = HierarchicalUtils.printtree(io, n)


export SumNode, TransformationNode, ProductNode
export densesharedmixture, nosharedmixture, allsharedmixture, priors, updatelatent!, buildmixture, pathcount, batchlogpdf
export em!, fit!

end # module
