push!(LOAD_PATH,"/home/ec2-user/julia/Pkg")
using ADatasets, SumDenseProduct, Flux, IterTools, StatsBase, Distributions, BenchmarkTools
using ADatasets: makeset, loaddataset, subsampleanomalous
using Flux: throttle, train!, Params
using EvalCurves, DrWatson, BSON

datasets = ["breast-cancer-wisconsin", "cardiotocography", "magic-telescope", "pendigits", "pima-indians", "wall-following-robot", "waveform-1", "waveform-2", "yeast"]
idir = filter(isdir,["/Users/tpevny/Work/Data/datasets/numerical","/mnt/output/data/datasets/numerical","/opt/output/data/datasets/numerical"])[1];
odir = filter(isdir,["/Users/tpevny/Work/Julia/results/datasets","/mnt/output/results/datasets","/opt/output/results/datasets"])[1];

include("exframework.jl")
dataset = "pendigits"
println(dataset," ", aparam)
aparam, dataparts, repetition = (type = "easy", polution = 0.0, variation = "low"), (0.8, 0.2), 1
trndata, tstdata, clusterdness = makeset(loaddataset(dataset, aparam.type ,idir)..., dataparts[1], aparam.variation, repetition)


n, l, σ, sharing = 8, 3, identity, :all

X = subsampleanomalous(trndata, aparam.polution)[1]

d = size(X, 1)
model = buildmixture(d, n, l, σ; sharing = sharing)
println("time to evaluate a single likelihood:")
@btime -mean(logpdf(model, X[:,1:100]))

println("time to calculate gradient:")
@btime gradient(() -> mean(logpdf(model, X[:,1:100])), Flux.params(model))

fit!(model, X, p.batchsize, p.steps; minimum_improvement = p.minimum_improvement, opt = ADAM(), debugfile = "debug_$(myid()).bson")
model, modelparams = fit(X)
BSON.@save oprefix*"_model.bson" model
stats = merge(supervisedstats(model, trndata, "train_"),
  supervisedstats(model, tstdata, "test_"),
  unsupervisedstats(model, X, "unsup_"),
  modelparams,
  (repetition = repetition, dataset = dataset)
  )
BSON.@save oprefix*"_stats.bson" stats aparam dataparts


  function randpars()
    modelparams = (batchsize = 100,
      steps = rand([5000, 10000, 20000]),
      minimum_improvement =  1e-2,
      n = rand([2, 4, 8, 16]),
      l = rand([1, 2, 3]),
      σ = rand([identity]),
      sharing = rand([:dense, :all, :none]))
  end

  function fit(X, p)
    n, l, σ, sharing = p.n, p.l, p.σ, p.sharing
    d = size(X, 1)
    model = buildmixture(d, n, l, σ; sharing = sharing)
    (fit!(model, X, p.batchsize, p.steps; minimum_improvement = p.minimum_improvement, opt = ADAM(), debugfile = "debug_$(myid()).bson"), p)
  end


  function runexp(dataset, repetition)
    modelparams = randpars()
    ofname = "ex1_"*savename(modelparams)*replace("_$(modelparams.σ)","NNlib." => "")
    isfile(joinpath(odir,dataset,"sumdense",ofname*"_model.bson")) && return(nothing)
    !isdir(joinpath(odir,dataset,"sumdense")) && mkpath(joinpath(odir,dataset,"sumdense"))
    anomalyexperiment(x -> fit(x, modelparams), dataset , joinpath(odir,dataset,"sumdense",ofname), aparam = (type = "easy", polution = 0.1, variation = "low"), repetition = repetition)
    return(nothing)
  end
end

pmap(p -> runexp(p[3], p[1]), Iterators.product(datasets, 1:5,1:100))


