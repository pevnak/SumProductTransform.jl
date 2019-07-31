using ToyProblems, Distributions, DenseMixtureModels, Unitary, Flux, FluxExtensions
using DenseMixtureModels: estep!, estep, DenseMixture, DenseP
using Unitary: SVDDense
using Plots
plotly()

x = eightskewedgaussians(200)

m = DenseMixture([DenseP(SVDDense(identity), MultivariateNormal(2,1)) for i in 1:8], fill(1/8, 8));
l = estep(m, x)
ps = Flux.params(m)
opt = ADAM()
Flux.train!(i -> -estep(m, x), Flux.Params(ps), 1:10000, opt; cb = () -> @show estep(m, x))
estep!(m, x)

xr = range(minimum(x[1,:]) - 1 , maximum(x[1,:])+ 1 , length = 100)
yr = range(minimum(x[2,:]) - 1 , maximum(x[2,:])+ 1 , length = 100)
contour(xr, yr, (x...) ->  Flux.data(logpdf(m, x)));
scatter!(x[1,:], x[2,:])

