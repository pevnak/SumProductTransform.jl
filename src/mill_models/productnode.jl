
Distributions.logpdf(m::ProductNode, x::Mill.ArrayNode) = logpdf(m, x.data)

# TO DO: Distributions.logpdf(m::ProductNode, x::Mill.ProductNode)