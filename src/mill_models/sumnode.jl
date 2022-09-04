
Distributions.logpdf(m::SumNode, x::ArrayNode) = logpdf(m, x.data)
Distributions.logpdf(m::SumNode, x::BagNode) = logpdf(m, x)

# TO DO: Distributions.logpdf(m::SumNode, x::ProductNode) 
