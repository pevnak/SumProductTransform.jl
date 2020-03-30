abstract type AbstractScope end

"""
struct Scope <: AbstractScope
	dims::Vector{Int}
	n::Int

	active::Vector{Bool}
	idxmap::Vector{Int}
end

The scoping through "Scope" supports to restrict the calculation 
of likelihood only to subset of random variables. The Scoping has 
to be supported by the the Transformation nodes and by lists.

For example below, we have distribution on four variables
```
s = Scope(4)
```
We can restrict the scope to dimensions [1,3] as 
```
julia> s = s[[1,3]]
Scope([1, 3], 4, Bool[1, 0, 1, 0], [1, 0, 2, 0])
```
and it can restricted even further to just third coordinate
```
julia> s[[2]]
Scope([3], 4, Bool[0, 0, 1, 0], [0, 0, 1, 0])
```
Notice that the indexing is translated, such that the scope 
with two active variables has only two indexes and they are 
translated to correct indexes.
"""
struct Scope <: AbstractScope
	dims::Vector{Int}
	n::Int

	active::Vector{Bool}
	idxmap::Vector{Int}
	function Scope(dims, n)
		active = falses(n)
		active[dims] .= true
		idxmap = zeros(n)
		idxmap[active] = 1:sum(active)
		new(dims, n, active, idxmap)
	end
end

struct FullScope <: AbstractScope
	n::Int
end

Scope(n::Int) = FullScope(n)

struct NoScope <: AbstractScope
end


active(s::Scope) = s.active
active(s::FullScope) = fill(true, s.n)

Base.getindex(s::FullScope, dims) = Scope(collect(1:s.n)[dims], s.n)
Base.getindex(s::Scope, dims) = Scope(s.dims[dims], s.n)
Base.getindex(s::NoScope, dims) = s


Base.filter(s::FullScope, θs, idxs) = θs, idxs
Base.filter(s::NoScope, θs, idxs) = θs, idxs
function Base.filter(s::Scope, θs, idxs)
	mask = map(i -> s.active[i[1]] && s.active[i[2]], idxs)
	(θs[mask], map(i -> (s.idxmap[i[1]], s.idxmap[i[2]]), idxs[mask]))
end


# Unitary.mulax(a::Butterfly, x::TransposedMatVec, m::Scope)  = _mulax(a.θs[m], a.idxs[m], x, 1)
# Unitary.mulax(a::TransposedMatVec, x::TransposedMatVec, m::Scope)  = _mulax(a.parent.θs[m], a.parent.idxs[m], x, 1)
