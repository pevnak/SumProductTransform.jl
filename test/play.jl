using ToyProblems, Distributions, DenseMixtureModels, Unitary, Flux
using DenseMixtureModels: estep!, estep, DenseMixture, DenseP
using Unitary: SVDDense
using Plots
plotly()

function visualize(m, x)
	xr = range(minimum(x[1,:]) - 1 , maximum(x[1,:])+ 1 , length = 100)
	yr = range(minimum(x[2,:]) - 1 , maximum(x[2,:])+ 1 , length = 100)
	contour(xr, yr, (x...) ->  Flux.data(logpdf(m, x)));
	scatter!(x[1,:], x[2,:])
end

function createmixture(n, p = () -> MultivariateNormal(2,1))
	DenseMixture([DenseP(SVDDense(identity), p()) for i in 1:n], fill(1/n, n))
end
###############################################################################
#			Let's do a single mixture
###############################################################################
x = eightskewedgaussians(200)
m = createmixture(8);
l = estep(m, x)
ps = Flux.params(m)
opt = ADAM()
Flux.train!(i -> -estep(m, x), Flux.Params(ps), 1:10000, opt; cb = () -> @show estep(m, x))
estep!(m, x)
visualize(m, x)



###############################################################################
#			Let's try two nested mixtures
###############################################################################
m = createmixture(4, () -> createmixture(2))
l = estep(m, x)
ps = Flux.params(m)
opt = ADAM()
Flux.train!(i -> -estep(m, x), Flux.Params(ps), 1:10000, opt; cb = () -> @show estep(m, x))
estep!(m, x)
visualize(m, x)

###############################################################################
#			Let's try shared inner mixture
###############################################################################
mi = createmixture(4);
m = createmixture(4, () -> mi)
ps = Flux.params(m)
opt = RMSProp(0.01)
Flux.train!(i -> -estep(m, x), Flux.Params(ps), 1:5000, opt; cb = () -> @show estep(m, x))
estep!(m, x) # -0.15319616593710528 (tracked)
visualize(m, x)

###############################################################################
#			Let's try shared more
###############################################################################
mmi = createmixture(3);
mi = createmixture(3, () -> mmi);
m = createmixture(3, () -> mi);
ps = Flux.params(m)
opt = RMSProp(0.01)
Flux.train!(i -> -estep(m, x), Flux.Params(ps), 1:5000, opt; cb = () -> @show estep(m, x))
estep!(m, x) 
visualize(m, x)
