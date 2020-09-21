_maptree(p::Distribution, x::AbstractMatrix) = (logpdf(p, x)[:], fill(tuple(), size(x,2)))
batchlogpdf(p, x, bs::Int) = reduce(vcat, map(i -> logpdf(p, x[:,i]), Iterators.partition(1:size(x,2), bs)))

"""
    pathcount(m)

    Number of possible path of a model. For distributions it defaults to one.
"""
pathcount(m) = 1

"""
    treelogpdf(p, x, path)

    logpdf of samples `x` calculated along the `path` determining components in sumnodes (at the moment)
    For distributions outside the SumProductTransform it falls back to logpdf(p, x).
"""
treelogpdf(m, x, path) = logpdf(m, x)
batchtreelogpdf(m, x, path) = map(i -> treelogpdf(m, x[:,i:i], path[i])[1], 1:length(path))

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
    path = samplepath(m)

    sample a path trough the model, which can be used by treelogpdf to calculate the
    pdf along this path.
"""
samplepath(m) = tuple()

####
#  compatibility with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:Distribution}) = LeafNode()
HierarchicalUtils.noderepr(node::T) where {T<:Distribution} = "$(T)"
