
using Serialization
"""
    anomalyexperiment(fit, trainstats, teststats, dataset; aparam, dataparts, repetition)

"""
function anomalyexperiment(fit, dataset, oprefix; aparam = (type = "easy", polution = 0.0, variation = "low"), dataparts = (0.8, 0.2), repetition = 1)
  println(dataset," ", aparam)
  oprefix *= savename(aparam)*"_$(repetition)"
  trndata, tstdata, clusterdness = makeset(loaddataset(dataset, aparam.type ,idir)..., dataparts[1], aparam.variation, repetition)

  X = subsampleanomalous(trndata, aparam.polution)[1]
  model, history, modelparams = fit(X)
  BSON.@save oprefix*"_model.bson" model
  serialize(oprefix*"_model.jls",model)
  stats = merge(supervisedstats(model, trndata, "train_"),
    supervisedstats(model, tstdata, "test_"),
    unsupervisedstats(model, X, "unsup_"),
    modelparams,
    (repetition = repetition, dataset = dataset)
    )
  BSON.@save oprefix*"_stats.bson" stats aparam dataparts
  serialize(oprefix*"_stats.jls", stats)
end

function reevaluate(dataset, modelfile; aparam = (type = "easy", polution = 0.0, variation = "low"), dataparts = (0.8, 0.2), repetition = 1)
  println(dataset," ", aparam)
  trndata, tstdata, clusterdness = makeset(loaddataset(dataset, aparam.type ,idir)..., dataparts[1], aparam.variation, repetition)

  X = subsampleanomalous(trndata, aparam.polution)[1]
  model = deserialize(modelfile)
  stats = merge(supervisedstats(model, trndata, "train_"),
    supervisedstats(model, tstdata, "test_"),
    unsupervisedstats(model, X, "unsup_"),
    (repetition = repetition, dataset = dataset)
    )
  oprefix = replace(modelfile, "_model.jls" => "")
  BSON.@save oprefix*"_stats.bson" stats aparam dataparts
  serialize(oprefix*"_stats.jls", stats)
end


"""
    supervisedstats(ŷ, y, prefix::String = "")
    supervisedstats(model, x, y::Vector{Int}, prefix::String = "")
    supervisedstats(model, data::Tuple, prefix::String = "")

    calculates supervised statistics from anomaly score `ŷ` and true labels `y` (1 --- normal, 2 --- anomaly)
      - density level at 1% and 5% from normal samples
      - likelihood from normal samples 
      - area under roc curve

    results are returned in NamedTuple, prefix can be added to names
"""
function supervisedstats(ŷ::Vector{T}, y::Vector{Int}, prefix::String = "") where {T<:Real}
  dld01, dld05 = quantile(ŷ[y .== 1], (0.01, 0.05))
  lkl = mean(ŷ[y .== 1])

  auc = EvalCurves.auc(EvalCurves.roccurve(-ŷ, y .- 1)...)
  names = tuple(map(p -> Symbol(prefix,p), ["dld01", "dld05", "lkl", "auc"])...)
  NamedTuple{names}([dld01, dld05, lkl, auc])
end
supervisedstats(model, x, y::Vector{Int}, prefix::String = "") = supervisedstats(batchlogpdf(model, x, 100), y, prefix)
supervisedstats(model, data::Tuple, prefix::String = "") = supervisedstats(model, data..., prefix)


"""
    unsupervisedstats(ŷ, prefix::String = "")

    calculates unsupervisedstats statistics from anomaly score `ŷ` 
      - density level at 1% and 5% from normal samples
      - likelihood from normal samples 

    results are returned in NamedTuple, prefix can be added to names
"""
function unsupervisedstats(ŷ::Vector{T}, prefix::String = "") where {T<:Real}
  dld01, dld05 = quantile(ŷ, (0.01, 0.05))
  lkl = mean(ŷ)
  names = tuple(map(p -> Symbol(prefix,p), ["dld01", "dld05", "lkl"])...)
  NamedTuple{names}([dld01, dld05, lkl])
end
unsupervisedstats(model, x, y) = unsupervisedstats(batchlogpdf(model, x, 100))
