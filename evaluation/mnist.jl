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
nc = [8,8,8]
model = buildmixture(d, nc,[identity, identity, identity], round.(Int,noise.*d));
tstidx = sample(1:size(X,2),100, replace = false)
xval = X[:,tstidx]

batchsize, iterations, maxpath = 100, 10000, 20
fit!(model, X, batchsize, iterations, maxpath, xval = xval)

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
# 1000: likelihood = -142.7921  time per iteration: 6.339553170382s likelihood time: 1.182214635
# 2000: likelihood = -7.361019  time per iteration: 6.253261801072s likelihood time: 1.184876745
# 3000: likelihood = 77.044945  time per iteration: 6.212154372596999s likelihood time: 1.178859881
# 4000: likelihood = 140.3892  time per iteration: 6.188441184737499s likelihood time: 1.179851148
# 5000: likelihood = 182.72734  time per iteration: 6.1707839420415995s likelihood time: 1.169764884
# 6000: likelihood = 222.26833  time per iteration: 6.163904881369166s likelihood time: 1.163184581
# 7000: likelihood = 218.73949  time per iteration: 6.167732035158285s likelihood time: 1.410927462