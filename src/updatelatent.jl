using MLDataPattern

"""
	updatelatent!(m, x, bs::Int = typemax(Int))

	estimate the probability of a component in `m` using data in `x`.
	if `bs < size(x,2)`, then the update is calculated part by part to save memory
"""
function updatelatent!(m, x, bs::Int = typemax(Int), prior_count = 1)
	ps = priors(m)
    foreach(p -> p .= prior_count, ps)
	paths = mappath(m, x)[2]
	foreach(p -> _updatelatent!(m, p), paths)
    foreach(normalizelatent!, ps)
end

_updatelatent!(m, path) = nothing

function normalizelatent!(w)
    w .= w ./ max(sum(w), 1) 
    w .= log.(w)
end

function mappath(m, x)
	n = nobs(x)
	mappath(m, x, collect(Iterators.partition(1:n, div(n,min(n,Threads.nthreads())))))
end

function mappath(m, x, segments::Vector)
    if length(segments) == 1
        return(_mappath(m, getobs(x, segments[1])))
    else 
        i = div(length(segments),2)
        s1, s2 = segments[1:i], segments[i+1:end]
        ref1 = Threads.@spawn mappath(m, x, s1)
        ref2 = Threads.@spawn mappath(m, x, s2)
        a1,a2 = fetch(ref1), fetch(ref2)
        return(vcat(a1[1],a2[1]), vcat(a1[2],a2[2]))
    end
end

