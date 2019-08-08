using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux
using Flux:throttle

using Plots
plotly()

function visualize(m, x)
	xr = range(minimum(x[1,:]) - 1 , maximum(x[1,:])+ 1 , length = 100)
	yr = range(minimum(x[2,:]) - 1 , maximum(x[2,:])+ 1 , length = 100)
	contour(xr, yr, (x...) ->  Flux.data(logpdf(m, x)));
	scatter!(x[1,:], x[2,:])
end


###############################################################################
#			Let's do a single mixture
###############################################################################
x = flower(200)
m = allsharedmixture(2, 8, 1)
# l = logpdf(m, x)
ps = Flux.params(m);
# gs = gradient(() ->  mean(logpdf(m, x)), ps)
opt = ADAM()
Flux.train!(i -> -mean(logpdf(m, x)), Flux.Params(ps), 1:10000, opt; cb = throttle(() -> (@show mean(logpdf(m, x))),10))
updatelatent!(m, x);
visualize(m, x)


###############################################################################
#			Let's try two nested mixtures
###############################################################################
m = nosharedmixture(2, 4, 2, identity, MultivariateNormal(2,1))
ps = Flux.params(m);
opt = ADAM()
Flux.train!(i -> -mean(logpdf(m, x)), Flux.Params(ps), 1:10000, opt; cb = throttle(() -> (@show mean(logpdf(m, x))),10))
updatelatent!(m, x);
m
visualize(m, x)

###############################################################################
#			Let's try shared inner mixture
###############################################################################
m = allsharedmixture(2, 4, 2, identity, MultivariateNormal(2,1))
ps = Flux.params(m);
opt = ADAM()
Flux.train!(i -> -mean(logpdf(m, x)), Flux.Params(ps), 1:5000, opt; cb = throttle(() -> (@show mean(logpdf(m, x))),10))
updatelatent!(m, x);
visualize(m, x)

###############################################################################
#			Let's try shared more
###############################################################################
mmi = createmixture(3);
mi = createmixture(3, identity, () -> mmi);
m = createmixture(3, identity,() -> mi);
ps = Flux.params(m);
opt = RMSProp(0.01)
Flux.train!(i -> -mean(logpdf(m, x)), Flux.Params(ps), 1:5000, opt; cb = throttle(() -> (@show mean(logpdf(m, x))),10))
updatelatent!(m, x); 
visualize(m, x)
