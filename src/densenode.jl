struct DenseNode{M,P}
	m::M
	p::P 
end

Base.length(m::DenseNode) = length(m.p)
Flux.@functor DenseNode


Distributions.logpdf(m::DenseNode, x::AbstractVector) = logpdf(m, reshape(x, :, 1))[1]
Distributions.logpdf(m::DenseNode, x::AbstractVector, s::NoScope) = logpdf(m, reshape(x, :, 1))[1]
function Distributions.logpdf(m::DenseNode, x::AbstractMatrix{T}) where {T}
	x, l = m.m((x,zero(T)))
	logpdf(m.p, x) .+ l[:]
end

function Distributions.logpdf(m::DenseNode, x::AbstractMatrix{T}, s::AbstractScope) where {T}
	x, l, _ = m.m((x, zero(T), s))
	logpdf(m.p, x, s) .+ l[:]
end

function pathlogpdf(m::DenseNode, x::AbstractMatrix{T}, path) where {T}
	s = path[1]
	x, l, _ = m.m((x, zero(T), s))
	pathlogpdf(m.p, x, path[2]) .+ l[:]
end

pathcount(m::DenseNode) = pathcount(m.p)
samplepath(m::DenseNode) = (NoScope(), samplepath(m.p))
samplepath(m::DenseNode, s::AbstractScope) = (s, samplepath(m.p, s))
updateprior!(ps::Priors, m::DenseNode, path) = updateprior!(ps, m.p, path)

function _mappath(m::DenseNode, x::AbstractMatrix{T}, s::AbstractScope = NoScope()) where {T}
	x, l = m.m((x, zero(T), s))
	lkl, path = _mappath(m.p, x, s)
	return(lkl .+ l[:], path)
end


Base.rand(m::DenseNode) = inv(m.m)(rand(m.p))

Base.show(io::IO, m::DenseNode{M,P}) where {M,P} = dsprint(io, m)
function dsprint(io::IO, n::DenseNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → ", color=c)
    dsprint(io, n.p, pad=[pad; (c, "     ")])
end
function dsprint(io::IO, n::DenseNode{M,P}; pad=[]) where {M,P<:MvNormal}
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, " $(n.m) → MvNormal\n", color=c)
end

