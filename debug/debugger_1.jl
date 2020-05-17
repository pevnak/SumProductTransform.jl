# push!(LOAD_PATH,"/home/ec2-user/julia/Pkg")
# using ProfileView, Profile
using ADatasets, SumDenseProduct, Flux, IterTools, StatsBase, Distributions, BenchmarkTools
using ADatasets: makeset, loaddataset, subsampleanomalous
using Flux: throttle, train!, Params
using EvalCurves, DrWatson, BSON

datasets = ["breast-cancer-wisconsin", "cardiotocography", "magic-telescope", "pendigits", "pima-indians", "wall-following-robot", "waveform-1", "waveform-2", "yeast"]
idir = filter(isdir,["/Users/tpevny/Work/Data/datasets/numerical","/mnt/output/data/datasets/numerical","/opt/output/data/datasets/numerical"])[1];
odir = filter(isdir,["/Users/tpevny/Work/Julia/results/datasets","/mnt/output/results/datasets","/opt/output/results/datasets"])[1];

# include("exframework.jl")

dataset = "pendigits"
aparam, dataparts, repetition = (type = "easy", polution = 0.0, variation = "low"), (0.8, 0.2), 1
println(dataset," ", aparam)
trndata, tstdata, clusterdness = makeset(loaddataset(dataset, aparam.type ,idir)..., dataparts[1], aparam.variation, repetition)


n, l, σ, sharing = 16, 1, identity, :all

X = subsampleanomalous(trndata, aparam.polution)[1]
d = size(X, 1)
model = buildmixture(d, n, l, σ; sharing = sharing);
t₁ = @elapsed -mean(logpdf(model, X[:,1:100]))
t₂ = @elapsed -mean(logpdf(model, X[:,1:100]))
println("time to evaluate a single likelihood: ", t₁, "  ",t₂)

t₁ = @elapsed gradient(() -> mean(logpdf(model, X[:,1:100])), Flux.params(model))
t₂ = @elapsed gradient(() -> mean(logpdf(model, X[:,1:100])), Flux.params(model))
println("time to evaluate a gradient likelihood: ", t₁, "  ",t₂)

x = X[:,1:100];
x1, x2 = X[:,1:50], X[:,51:100]

ps = Flux.params(model)
batchsize, maxtree = 100, 100

SumDenseProduct.tunegrad(model, x, batchsize, maxtree, ps)


besttree = [SumDenseProduct.sampletree(model) for i in 1:size(X,2)]
SumDenseProduct.samplinggrad(model, X, besttree, batchsize, maxtree, ps)
# fit!(model, X, p.batchsize, p.steps; minimum_improvement = p.minimum_improvement, opt = ADAM(), debugfile = "debug_$(myid()).bson")

# ProfileView.svgwrite("/tmp/profile_results.svg")