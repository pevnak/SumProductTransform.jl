using Zygote: dropgrad

struct DenseMixture{T,C}
	components::Vector{C}
	prior::Vector{T}
end

Base.show(io::IO, z::DenseMixture{T,C}) where {T,C} = print(io, "DenseMixture with $(length(z.components)) components")

Flux.children(x::DenseMixture) = x.components
Flux.mapchildren(f, x::DenseMixture) = f.(Flux.children(x))

function createmixture(n, σ = identity, p = () -> MultivariateNormal(2,1))
	DenseMixture([DenseP(Unitary.SVDDense(σ), p()) for i in 1:n], fill(1/n, n))
end

Zygote.@adjoint Flux.onehotbatch(y, n) = Flux.onehotbatch(y,n), Δ -> (nothing, nothing)
Zygote.@adjoint Flux.onecold(x) = Flux.onecold(x), Δ -> (nothing,)


"""
	logpdf(m::MixtureModel, x)

	updates the prior distribution of components estimated from data `x` given components
	and returns likelihood assuming the most likely component as the generating component
"""
function Distributions.logpdf(m::DenseMixture, x)
	lkl = transpose(hcat(map(c -> logpdf(c, x),m.components)...))
	if Flux.istraining()
		y = Flux.onecold(softmax(dropgrad(lkl), dims = 1))
		o = Flux.onehotbatch(y, 1:length(m.components))
		# o = softmax(lkl, dims = 1)
		return(mean( o .* lkl, dims = 1)[:])
	else
		return(logsumexp(log.(m.prior .+ 1f-8) .+ lkl, dims = 1)[:])
	end
end

function updateprior!(m::DenseMixture, x)
	zeroprior!(m);
	_updateprior!(m::DenseMixture, x);
	normalizeprior!(m);
end

zeroprior!(m) = nothing
function zeroprior!(m::DenseMixture)
	m.prior .= 0 
	foreach(zeroprior!, m.components)
	nothing
end

normalizeprior!(m) = nothing
function normalizeprior!(m::DenseMixture)
	m.prior ./= max(sum(m.prior), 1)  
end

function _updateprior!(m::DenseMixture, x)
	lkl = transpose(hcat(map(c -> logpdf(c, x),m.components)...))
	y = Flux.onecold(softmax(dropgrad(lkl), dims = 1))
	o = Flux.onehotbatch(y, 1:length(m.components))
	m.prior .+= sum(o, dims = 2)[:]
end

Distributions.logpdf(m::DenseMixture, x::Tuple) = logpdf(m, reshape(collect(x), :, 1))[1]