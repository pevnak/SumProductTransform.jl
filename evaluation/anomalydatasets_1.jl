using Distributed, ArgParse

# s = ArgParseSettings()
# @add_arg_table s begin
#     ("--variant"; nargs = '*'; default=["fixed"]);
#     ("--dataset"; nargs = '*'; default=[]);
# end
# settings = parse_args(ARGS, s; as_symbols=true)

@everywhere begin
  push!(LOAD_PATH,"/home/ec2-user/julia/Pkg")
  using ADatasets, SumDenseProduct, Flux, IterTools, StatsBase, Distributions
  using ADatasets: makeset, loaddataset, subsampleanomalous
  using Flux: throttle, train!, Params
  using EvalCurves, DrWatson, BSON

  datasets = ["breast-cancer-wisconsin", "cardiotocography", "magic-telescope", "pendigits", "pima-indians", "wall-following-robot", "waveform-1", "waveform-2", "yeast"]
  idir = filter(isdir,["/Users/tpevny/Work/Data/datasets/numerical","/mnt/output/data/datasets/numerical","/opt/output/data/datasets/numerical"])[1];
  odir = filter(isdir,["/Users/tpevny/Work/Julia/results/datasets","/mnt/output/results/datasets","/opt/output/results/datasets"])[1];

  include("exframework.jl")


  function randpars()
    modelparams = (batchsize = 100,
      steps = rand([5000, 10000, 20000]),
      minimum_improvement =  1e-2,
      n = rand([2, 4, 8, 16]),
      l = rand([1, 2, 3, 4]),
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
    ofname = "ex1_"*savename(modelparams)*replace("$(modelparams.σ)","NNlib." => "")
    isfile(joinpath(odir,dataset,"sumdense",ofname*"_model.bson")) && return(nothing)
    !isdir(joinpath(odir,dataset,"sumdense")) && mkpath(joinpath(odir,dataset,"sumdense"))
    anomalyexperiment(x -> fit(x, modelparams), dataset , joinpath(odir,dataset,"sumdense",ofname), aparam = (type = "easy", polution = 0.1, variation = "low"), repetition = repetition)
    return(nothing)
  end
end

pmap(p -> runexp(p[3], p[1]), Iterators.product(1:5,1:100, datasets))[]


