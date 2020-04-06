using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, LinearAlgebra, SpecialFunctions
using SparseArrays, StatsBase
using SumDenseProduct: logsumexp
using Flux:throttle
using Plots
include("distributions.jl")

Ndat = 200;
x = flower(Float32,Ndat);
K = 19
comps = tuple([DenseNode(Unitary.SVDDense(2, identity, :butterfly), MvNormal(2,1f0)) for _ in 1:K]...)
α₀ = fill(1f0, K)
ρ = 1e-5*ones(K,Ndat)
α = deepcopy(α₀)
ps = Flux.params(comps)
opt = ADAM()
ϕρ = 0.9;
ϕα = 0.999;

# model = SumNode(collect(comps))
# fit!(model, x, 100, 10000, 0; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())

l(x,y)=logpdf(SumNode(collect(comps)),[x;y])[1]
contour(-10.0:0.1:10,-10.0:0.1:10,l,levels=50)

Nci = 5;
Nxi = 50;
ci = Int.(1:Nci)
xi = Int.(1:Nxi)
niter = 1000;
loss(ri,xind,cind) = - sum(ri[cind,:] .* hcat(map(c->logpdf(c, x[:,xind]),comps[cind])...)')
for i in 1:niter
	global α, comps, opt, ρ, ci, xi
	ci .= Int.(round.((K-1)*rand(Nci)).+1)
	xi .= Int.(round.((Ndat-1)*rand(Nxi)).+1)

	#update statistics
    ρi = vcat(map(c->Matrix(logpdf(c, x[:,xi])'),comps[ci])...)
	ρi .+= log.(α[ci])
	if false
		ρ[ci,xi[:]] = ϕρ*ρ[ci,xi]+(1-ϕρ)*ρi
		eρi = exp.(ρ[:,xi])
		ri = eρi ./ sum(eρi, dims = 1)
		α =  ϕα*α+(1-ϕα)*sum(ri,dims=2)[:]
	else
		ρ[ci,xi[:]] = ϕρ*ρ[ci,xi[:]]+ρi
		eρi = exp.(ρ[:,xi[:]])
		ri = eρi ./ sum(eρi, dims = 1)
		α =  ϕα*α+sum(ri,dims=2)[:]
	end

	gs = gradient(() -> loss(ri,xi,ci) , ps)
	# gs = gradient(() -> - sum(logsumexp(ed .+ r .+ hcat(map(c -> logpdf(c, x), comps)...), dims = 2)), ps)
	Flux.Optimise.update!(opt, ps, gs)
	mod(i, 1000) == 0 && @show mean(log_likelihood(comps, α, x))
end

