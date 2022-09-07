
Distributions.logpdf(m::SumNode, x::ArrayNode) = logpdf(m, x.data)

function Distributions.logpdf(m::SumNode, x::BagNode)
	lkl = transpose(hcat(map(c -> logpdf(c, x) ,m.components)...))
	w = m.prior .- logsumexp(m.prior)
	logsumexp(w .+ lkl, dims = 1)[:]
end

Base.rand(m::SumNode, n::Integer) = rand(m.components[sample(Weights(m.prior))], n)

# TO DO: Distributions.logpdf(m::SumNode, x::ProductNode) 
