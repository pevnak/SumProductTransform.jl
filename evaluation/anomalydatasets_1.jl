using Distributed, ArgParse
include("setpath.jl")
s = ArgParseSettings()
@add_arg_table s begin
    ("--repetition"; arg_type = Int; default=1);
    ("--polution"; arg_type = Float64; default=0.0);
    ("--dataset"; nargs = '*'; default=[]);
end
settings = parse_args(ARGS, s; as_symbols=true)

@everywhere begin
  using ADatasets, SumDenseProduct, Flux, IterTools, StatsBase, Distributions, Unitary
  using ADatasets: makeset, loaddataset, subsampleanomalous
  using Flux: throttle, train!, Params
  using EvalCurves, DrWatson, BSON

  datasets = ["breast-cancer-wisconsin", "cardiotocography", "magic-telescope", "pendigits", "pima-indians", "wall-following-robot", "waveform-1", "waveform-2", "yeast"]
  idir = filter(isdir,["/Users/tpevny/Work/Data/datasets/numerical","/home/pevnytom/Data/numerical"])[1];
  odir = filter(isdir,["/Users/tpevny/Work/Julia/results/datasets","/home/pevnytom/Data/results/datasets"])[1];

  include("exframework.jl")

  function randpars()
    modelparams = (batchsize = 100,
      maxtree = rand([50,100,200]),
      firsttransform = rand([true,false]),
      steps = rand([5000, 10000, 20000]),
      minimum_improvement =  1e-2,
      n = rand([2, 4, 8, 16]),
      l = rand([1, 2, 3, 4]),
      σ = rand([identity]),
      sharing = rand([:transform, :all, :none]),
      gradmethod = rand([:sampling, :bestsampling, :exactpath]))
  end

  function fit(X, p)
    d = size(X, 1)
    model = buildmixture(d, p.n, p.l, p.σ; sharing = p.sharing, firsttransform = p.firsttransform)
    history = fit!(model, X, p.batchsize, p.steps, p.maxtree; gradmethod = p.gradmethod, minimum_improvement = p.minimum_improvement, opt = ADAM())
    (model, p)
  end


  function runexp(dataset, repetition, polution)
    modelparams = randpars()
    ofname = "ex1_"*savename(modelparams)*replace("_$(modelparams.σ)","NNlib." => "")
    isfile(joinpath(odir,dataset,"sumtransform",ofname*"_model.bson")) && return(nothing)
    !isdir(joinpath(odir,dataset,"sumtransform")) && mkpath(joinpath(odir,dataset,"sumtransform"))
    anomalyexperiment(x -> fit(x, modelparams), dataset , joinpath(odir,dataset,"sumtransform",ofname), aparam = (type = "easy", polution = polution, variation = "low"), repetition = repetition)
    return(nothing)
  end
end

settings[:dataset] = isempty(settings[:dataset]) ? datasets : settings[:dataset]
map(p -> runexp(p[1], settings[:repetition]), Iterators.product(settings[:dataset],1:100))


