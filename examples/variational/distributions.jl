using SpecialFunctions
using Zygote

Zygote.@nograd rand

function e_dirichlet(α, k)
	digamma(α[k] + eps(eltype(α))) - digamma(sum(α))
end

e_dirichlet(α) = [e_dirichlet(α, k) for k in 1:K]

function kldir(α, β)
	loggamma(sum(α)) - loggamma(sum(β)) - sum(loggamma.(α)) + sum(loggamma.(β)) + sum((α .- β) .* (digamma.(α) .- digamma(sum(α))));
end

function softmax_log_likelihood(components, α, x)
	w = α .- logsumexp(α, dims = 1)
	lkl = transpose(hcat(map(c -> logpdf(c, x) ,components)...))
	logsumexp(w .+ lkl, dims = 1)[:]
end

function sample_log_likelihood(components, α, x)
	π = rand(Distributions.Dirichlet(α))
	logπ = Matrix(transpose(log.(π)))
	logsumexp(logπ .+ hcat(map(c -> logpdf(c, x), components)...), dims = 2)
end


function log_likelihood(components, α, x, r = 100) 
	mean(sample_log_likelihood(components, α, x) for _ in 1:100)
end

function samplez(α)
	pii = rand(Distributions.Dirichlet(α))
	sample(Weights(pii))
end

function samplez(α, n)
	z = sparse(1:n, [samplez(α) for _ in 1:n], true, n, length(α))
end

function ∇log_samplez(α, z)
	α₀ = sum(α)
	n = size(z, 1)
	nk = sum(z, dims = 1)[:]
	digamma(α₀) .- digamma(n + α₀) .+ digamma.(nk .+ α) .- digamma.(α) 
end

function ∇log_samplezi(α, z)
	α₀ = sum(α)
	α = transpose(α)
	digamma(α₀) .- digamma(1 + α₀) .+ digamma.(z .+ α) .- digamma.(α) 
end


function sample_concrete(logα::Vector, τ, n)
	u = rand(Gumbel(), length(logα), n)
	x = (logα .+ u) ./ τ
	softmax(x)
end

function sample_concrete(logα::Matrix, τ)
	u = rand(Gumbel(), size(logα)...)
	x = (logα .+ u) ./ τ
	softmax(x)
end

function hard_max(x; dims)
	i = mapslices(argmax, x, dims = dims)[:]
	sparse(i, 1:size(x,2), true, size(x)...)
end

Zygote.@adjoint function hard_max(x; dims)
	i = mapslices(argmax, x, dims = dims)[:]
	o = sparse(i, 1:size(x,2), true, size(x)...)
	o, Δ -> (Δ .* o, nothing)
end