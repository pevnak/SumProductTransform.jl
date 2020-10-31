_maptree(p::Distribution, x::AbstractMatrix) = (logpdf(p, x)[:], fill(tuple(), size(x,2)))
batchlogpdf(p, x, bs::Int) = reduce(vcat, map(i -> logpdf(p, x[:,i]), Iterators.partition(1:size(x,2), bs)))

"""
    pathcount(m)

    Number of possible trees of a model. For distributions it defaults to one.
"""
treescount(m) = 1

"""
    treelogpdf(p, x, trees)

    logpdf of samples `x` calculated along the `trees` determining components in sumnodes (at the moment)
    For distributions outside the SumProductTransform it falls back to logpdf(p, x).
"""
treelogpdf(m, x, trees) = logpdf(m, x)
batchtreelogpdf(m, x, trees) = map(i -> treelogpdf(m, x[:,i:i], trees[i]), 1:length(trees))[1]

_priors(m) = nothing
function priors!(ps, x, seen = Flux.IdSet())
  x in seen && return
  push!(seen, x)
  xx = _priors(x)
  !isnothing(xx) && push!(ps, xx)
  for child in Flux.trainable(x)
    priors!(ps, child, seen)
  end
end

function priors(m)
  ps = []
  priors!(ps, m)
  return ps
end

"""
    trees = sampletree(m)

    sample a trees trough the model, which can be used by treelogpdf to calculate the
    pdf along this trees.
"""
sampletree(m) = tuple()

####
#  compatibility with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:Distribution}) = LeafNode()
HierarchicalUtils.noderepr(node::T) where {T<:Distribution} = "$(T)"
