"""
struct ProductNode
	components::T
	dimensions::U
end

	ProductNode implements a product of independent random variables. Each random 
	variable(s) can be of any type, which implements the interface of `Distributions`
	package (`logpdf` and `length`). Recall that `length` in case of distributions is 
	the dimension of a samples.
"""
struct ProductNode{T<:Tuple,U<:NTuple{N,UnitRange{Int}} where N}
	components::T
	dimensions::U
end

Flux.@functor ProductNode
Flux.trainable(m::ProductNode) = (m.components,)

"""
	ProductNode(ps::Tuple)

	ProductNode with `ps` independent random variables. Each random variable has to 
	implement `logpdf` and `length`.
"""
function ProductNode(ps::Tuple)
	dimensions = Vector{UnitRange{Int}}(undef, length(ps))
	start = 1
	for (i, p) in enumerate(ps)
		l = length(p)
		dimensions[i] = start:start + l - 1
		start += l 
	end
	ProductNode(ps, tuple(dimensions...))
end

function Distributions.logpdf(m::ProductNode, x, s::AbstractScope = NoScope())
	m
	o = logpdf(m.components[1], x[m.dimensions[1],:], s[m.dimensions[1]])
	for i in 2:length(m.components)
		o += logpdf(m.components[i], x[m.dimensions[i],:], s[m.dimensions[1]])
	end
	o
end

function pathlogpdf(p::ProductNode, x, path, s::AbstractScope = NoScope())
	o = pathlogpdf(p.components[1], x[p.dimensions[1],:], path[1], s[m.dimensions[1]])
	for i in 2:length(p.components)
		o += pathlogpdf(p.components[i], x[p.dimensions[i],:], path[i], s[m.dimensions[1]])
	end
	o
end

pathcount(m::ProductNode) = mapreduce(n -> pathcount(n), *, m.components)
samplepath(m::ProductNode) = map(samplepath, m.components)
function _updatelatent!(m::ProductNode, path)
	for i in 1:length(m.components)
		_updatelatent!(m.components[i], path[i])
	end
end


function _mappath(m::ProductNode, x)
	o, path = _mappath(m.components[1], x[m.dimensions[1],:])
	path = map(s -> (s,), path)
	for i in 2:length(m.components)
		oo, pp = _mappath(m.components[i], x[m.dimensions[i],:])
		o .+= oo
		path = map(s -> tuple(s[1]..., s[2]), zip(path, pp))
	end
	o, path
end

Base.rand(m::ProductNode) = vcat([rand(p) for p in m.components]...)


Base.length(m::ProductNode) = m.dimensions[end].stop
Base.getindex(m::ProductNode, i...) = getindex(m.components, i...)


Base.show(io::IO, z::ProductNode) = dsprint(io, z)
function dsprint(io::IO, n::ProductNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "Product\n", color=c)

    m = length(n.components)
    for i in 1:(m-1)
    	if typeof(n.components[i]) <: Distributions.MvNormal
    		paddedprint(io, "  ├── MvNormal\n", color=c, pad=pad)
    	else
	        paddedprint(io, "  ├── ", color=c, pad=pad)
	        dsprint(io, n.components[i], pad=[pad; (c, "  │   ")])
	    end
    end
	if typeof(n.components[end]) <: Distributions.MvNormal
	    paddedprint(io, "  └──  MvNormal\n", color=c, pad=pad)
	else
	    paddedprint(io, "  └── ", color=c, pad=pad)
	    dsprint(io, n.components[end], pad=[pad; (c, "      ")])
	end
end