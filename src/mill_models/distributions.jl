
Distributions.logpdf(d::UnivariateDistribution, x::ArrayNode) = logpdf(d, x.data)
Distributions.logpdf(d::MultivariateDistribution, x::ArrayNode) = logpdf(d, x.data)