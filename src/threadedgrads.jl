
function addgrad!(gs1, gs2, ps)
    for p in ps 
        if gs1[p] != nothing && gs2[p] != nothing 
            gs1.grads[p] .+= gs2[p]
        elseif gs2[p] != nothing
            gs1.grads[p] = gs2[p]
        end
    end
    gs1
end

function threadedgrad(f, ps, segments::Vector)
    if length(segments) == 1
        return(gradient(() -> f(segments[1]), ps))
    else 
        i = div(length(segments),2)
        s1, s2 = segments[1:i], segments[i+1:end]
        ref1 = Threads.@spawn threadedgrad(f, ps, s1)
        ref2 = Threads.@spawn threadedgrad(f, ps, s2)
        return(addgrad!(fetch(ref1), fetch(ref2), ps))
    end
end

function threadedgrad(f, ps, n::Int)
	segments = collect(Iterators.partition(1:n, div(n,Threads.nthreads())))
	gs = threadedgrad(f, ps, segments)
    for p in ps 
        gs[p] == nothing && continue
        gs[p] ./=n
    end
    gs
end