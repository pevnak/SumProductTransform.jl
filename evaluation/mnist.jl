using Flux, Flux.Data.MNIST, SumDenseProduct, IterTools, StatsBase, Distributions, Unitary
using Images
using Base.Iterators: partition
using Flux: throttle, train!, Params
using EvalCurves, DrWatson, BSON
# using Plots

# plotly();


# showimg(x) = heatmap(reshape(x, 28,28))

imgs = MNIST.images();
imgs = map(x -> imresize(x, ratio = 1/2), imgs);
X = Float32.(float(hcat(vec.(imgs)...)))
# showimg(data[1][:,1])
d = size(X,1)
noise = [0.5, 0.25, 0]
# noise = [0, 0 , 0]
nc = [10,10,10]
model = buildmixture(d, nc,[identity, identity, identity], round.(Int,noise.*d));
tstidx = sample(1:size(X,2),100, replace = false)
xval = X[:,tstidx]

batchsize, iterations, maxpath = 100, 10000, 100
SumDenseProduct.tunegrad(model, X, batchsize, maxpath, ps)
fit!(model, X, batchsize, iterations, maxpath, xval = xval);

# bestpdf, bestidxs = SumDenseProduct.samplepdf(model, xval, [3,3,3], 10)
# fit!(model, X, 100, 10000, xval = xval, debugfile = "/tmp/debug.bson")

# model = allsharedmixture(d, [10,10, 10],[identity, identity, identity], round.(Int,noise.*d));
# 200 butterflys 22.705610933
# 50 butterflys 18.681631758
# @elapsed mean(logpdf(model, xval))

# 2 components
# 1000: likelihood = -127.04439  time per iteration: 0.682315185392s likelihood time: 0.068306023
# model = model[1].c
# ps = Flux.params(model);
# @elapsed gs = gradient(() -> mean(logpdf(model, xval)), ps)

# d = 196
# julia> fit!(model, X, 100, 10000, nc, 100, xval = xval)
# 1000: likelihood = -147.48341  time per iteration: 4.381122604701s likelihood time: 1.160839419
# 2000: likelihood = -3.7913644  time per iteration: 4.367255536410999s likelihood time: 1.191263288
# 3000: likelihood = 90.979004  time per iteration: 4.331746652230333s likelihood time: 1.141525152
# 4000: likelihood = 161.15584  time per iteration: 4.24704645124475s likelihood time: 1.0936102
# 5000: likelihood = 210.47264  time per iteration: 4.1943532845048s likelihood time: 1.083334396
# 6000: likelihood = 249.8404  time per iteration: 4.160388583953334s likelihood time: 1.099181995
# 7000: likelihood = 279.53485  time per iteration: 4.1359216489634285s likelihood time: 1.073394427
# 8000: likelihood = 297.33212  time per iteration: 4.117470572517125s likelihood time: 1.222405512
# 9000: likelihood = 311.60693  time per iteration: 4.1031731320837785s likelihood time: 1.114199775
# 10000: likelihood = 264.3805  time per iteration: 4.092185192862801s likelihood time: 1.068880428