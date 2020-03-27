using MLDataPattern

struct Priors
    priors::IdDict{Any,Any}
end
Priors() = Priors(IdDict{Any,Any}())
Base.setindex!(p::Priors, v, k) = p.priors[k] = v
Base.getindex(p::Priors, k) = p.priors[k]
function Base.get(p::Priors, k, v) 
    haskey(p.priors, k) && return(p[k])
    p[k] = v
    v 
end
Base.iterate(p::Priors) = iterate(p.priors)
Base.iterate(p::Priors, s ) = iterate(p.priors, s)
Base.length(p::Priors) = length(p.priors)

"""
    updatepriors!(m, pr::Priors)
	updatepriors!(m, x, s::AbstractScope = NoScope(); prior_count = 1)

	overwrite priors from `pr`
    estimate priors of components in SumNodes by hard-em from `x` using 
    scoping `s`
"""
function updatepriors!(m, pr::Priors)
    for (k, w) in pr.priors 
        k .= w 
    end
end

function updatepriors!(m, x, s::AbstractScope = NoScope(); prior_count = 1)
    pr = calculatepriors(m, x, s)
    for p in values(pr.priors)
        p .+= prior_count
        normalizeprior!(p)
    end
    updatepriors!(m, pr)
end

@deprecate updatelatent!(m, x) updatepriors!(m, x)

"""
    calculatepriors(m, x, s::AbstractScope = NoScope())
    estimate priors of components in SumNodes by hard-em from `x` using 
    scoping `s`
"""
function calculatepriors(m, x, s::AbstractScope = NoScope())
    pr = Priors()
    paths = mappath(m, x, s)[2]
    foreach(path -> updateprior!(pr, m, path), paths)
    pr
end

_updatelatent!(m, path) = nothing
updateprior!(ps::Priors, m, path) = nothing

function normalizeprior!(w)
    w .= w ./ max(sum(w), 1) 
    w .= log.(w)
end

function mappath(m, x, s::AbstractScope)
	n = nobs(x)
	mappath(m, x, s, collect(Iterators.partition(1:n, div(n,min(n,Threads.nthreads())))))
end

function mappath(m, x, s::AbstractScope, segments::Vector)
    if length(segments) == 1
        return(_mappath(m, getobs(x, segments[1]), s))
    else 
        i = div(length(segments),2)
        s1, s2 = segments[1:i], segments[i+1:end]
        ref1 = Threads.@spawn mappath(m, x, s, s1)
        ref2 = Threads.@spawn mappath(m, x, s, s2)
        a1,a2 = fetch(ref1), fetch(ref2)
        return(vcat(a1[1],a2[1]), vcat(a1[2],a2[2]))
    end
end

