abstract type AbstractScope end

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
