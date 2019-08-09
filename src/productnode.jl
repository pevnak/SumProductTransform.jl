struct ProductNode{T<:Tuple,U<:NTuple{N,UnitRange{Int}} where N}
	components::T
	dimensions::U
end

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

Base.show(io::IO, z::ProductNode) = dsprint(io, z)
function dsprint(io::IO, n::ProductNode; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "Product\n", color=c, pad = pad)

    m = length(n.components)
    for i in 1:(m-1)
    	if typeof(n.components[i]) <: Distributions.MvNormal
    		paddedprint(io, "  ├── MvNormal\n", color=c, pad=pad)
    	else
	        paddedprint(io, "  ├── \n", color=c, pad=pad)
	        dsprint(io, n.components[i], pad=[pad; (c, "  │   ")])
	    end
    end
	if typeof(n.components[end]) <: Distributions.MvNormal
	    paddedprint(io, "  └──  MvNormal\n", color=c, pad=pad)
	else
	    paddedprint(io, "  └── \n", color=c, pad=pad)
	    dsprint(io, n.components[end], pad=[pad; (c, "      ")])
	end
end

Base.getindex(m::ProductNode, i...) = getindex(m.components, i...)

Flux.@treelike(ProductNode)
# Flux.children(x::ProductNode) = x.components
# Flux.mapchildren(f, x::ProductNode) = f.(Flux.children(x))

Distributions.logpdf(m::ProductNode, x) = sum(map( p -> logpdf(p[1], x[p[2],:]), zip(m.components, m.dimensions)))
Base.length(m::ProductNode) = m.dimensions[end].stop