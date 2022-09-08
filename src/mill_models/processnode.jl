
struct ProcessNode{T, S}
    feature::T
    cardinality::S
end

Flux.@functor ProcessNode

Base.length(m::ProcessNode) = length(m.feature)


function Distributions.logpdf(m::ProcessNode, x::Mill.BagNode)

    lpf_inst = logpdf(m.feature, x.data)

    ll = map(ids -> sum(lpf_inst[ids]) + logpdf(m.cardinality, length(ids)) + logfactorial(length(ids)), x.bags) # nefunguje
    # ll = map(ids -> sum(lpf_inst[ids]), x.bags)  # funguje
    # ll = map(ids -> sum(lpf_inst[ids]) + logfactorial(length(ids)), x.bags) # nefunguje
    # ll = map(ids -> sum(lpf_inst[ids]) + logfactorial(length(ids)), x.bags) # nefunguje
    # ll = map(ids -> sum(lpf_inst[ids]) + logpdf(m.cardinality, 3) + logfactorial(3), x.bags) # funguje
    

    
    # f_bag(bagid::Integer) = reduce(+, logpdf(m.feature, x[bagid].data); init = 0) # init for empty bag
    # logp_f = map(f_bag, collect(1:length(x.bags)))
    # x.data[:, bag]

    # mapreduce( i -> logpdf(m.feature, x.data[i], +, x.bags); init=0)
    # zip(x.bags, x)


    # ll = logp_c .+ logp_f .+ logfactorial.(card)
    return ll
end



####
#	Functions for sampling the model
####
Base.rand(m::ProcessNode) = rand(m.feature, rand(m.cardinality))
Base.rand(m::ProcessNode, n::Integer) = [rand(m.feature, rand(m.cardinality)) for _ in 1:n]


####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:ProcessNode}) = InnerNode()
HierarchicalUtils.noderepr(::ProcessNode) = "ProcessNode"
HierarchicalUtils.printchildren(node::ProcessNode) = (node.feature, node.cardinality)
