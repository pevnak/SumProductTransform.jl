using Flux, Flux.Data.MNIST, SumDenseProduct, IterTools, StatsBase, Distributions, Unitary
using Images
using Base.Iterators: partition
using Flux: throttle, train!, Params
using EvalCurves, DrWatson, BSON
using SumDenseProduct: samplepath
# using Plots

# plotly();


# showimg(x) = heatmap(reshape(x, 28,28))

imgs = MNIST.images();
imgs = map(x -> imresize(x, ratio = 1/4), imgs);
X = Float32.(float(hcat(vec.(imgs)...)))
# showimg(data[1][:,1])
d = size(X,1)
noise = [0.25, 0.25, 0.25, 0]
# noise = [0, 0 , 0]
nc = [20,20,20, 20]
model = buildmixture(d, nc,fill(identity, length(nc)), round.(Int,noise.*d));
tstidx = sample(1:size(X,2),100, replace = false)
xval = X[:,tstidx]

batchsize, iterations, maxpath = 100, 20000, 100
using BenchmarkTools
bestpath = [samplepath(model) for i in 1:size(xval,2)]
@btime SumDenseProduct.samplepdf!(bestpath, model, xval, maxpath)
# SumDenseProduct.tunegrad(model, X, batchsize, maxpath, Flux.params(model))
fit!(model, X, batchsize, iterations, maxpath, xval = xval);

#no multithreadding, 4 layers
# compilation of samplinggrad: 63.307045557
# execution of samplinggrad: 8.084872945
# compilation of exactgrad: 1020.45304096
# execution of exactgrad: 1025.862231723
# compilation of exactpathgrad: 41.461369366
# execution of exactpathgrad: 16.150681999

# compilation of samplinggrad: 7.402696915
# execution of samplinggrad: 6.901650627
# compilation of exactgrad: 37.060813949
# execution of exactgrad: 37.40641376
# compilation of exactpathgrad: 6.63332085
# execution of exactpathgrad: 6.377339665
# using exactpathgrad for calculation of the gradient
# d = 196

# model 10, 10, 10 9000: likelihood = 310.09567  time per iteration: 4.597955989184666s likelihood time: 1.037649129