log_normal(x) = - sum(x.^2, dims=1) / 2 .- size(x,1)*log(Float32(2π)) / 2
log_normal(x,μ) = log_normal(x .- μ)
log_normal(x,μ, σ2::T) where {T<:Number} = - sum((@. ((x - μ)^2)/σ2), dims=1)/2 .- size(x,1)*log(σ2*2π)/2

#Let's do a little bit of function stealing
Distributions.logpdf(p::MvNormal, x::AbstractMatrix) = log_normal(x)[:]
function Distributions.logpdf(m::M, x::AbstractMatrix) where {M<: MvNormal{T,Distributions.PDMats.ScalMat{T},FillArrays.Zeros{T,1,Tuple{Base.OneTo{Int64}}}}} where {T}
  log_normal(x, m.μ)[:]
end
_maptree(p::MvNormal, x::AbstractMatrix) = (logpdf(p, x)[:], fill(tuple(), size(x,2)))
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
HierarchicalUtils.NodeType(::Type{<:MvNormal}) = LeafNode()
HierarchicalUtils.noderepr(node::MvNormal) = "normal"


