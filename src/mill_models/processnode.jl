
mutable struct ProcessNode{T, S}
    feature::T
    cardinality::S
end

Flux.@functor ProcessNode

Base.length(m::ProcessNode) = length(m.feature)


function Distributions.logpdf(m::ProcessNode, x::BagNode)
    card = length.(x.bags)

    logp_c = logpdf.(m.cardinality, card)
    
    f_bag(idx) = reduce(+, logpdf(m.feature, x[idx].data); init = 0) # init for empty bag
    logp_f = map(f_bag, 1:length(x.bags))

    ll = logp_c .+ logp_f .+ logfactorial.(card)
    return ll
end



####
#	Functions for sampling the model
####
Base.rand(m::ProcessNode) = rand(m.feature, rand(m.cardinality))


####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:ProcessNode}) = InnerNode()
HierarchicalUtils.noderepr(::ProcessNode) = "ProcessNode"
HierarchicalUtils.printchildren(node::ProcessNode) = (node.feature, node.cardinality)
