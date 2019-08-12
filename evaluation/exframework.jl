"""
    anomalyexperiment(fit, trainstats, teststats, dataset; aparam, dataparts, repetition)

"""
function anomalyexperiment(fit, dataset, oprefix; aparam = (type = "easy", polution = 0.0, variation = "low"), dataparts = (0.8, 0.2), repetition = 1)
  println("parameters of anomalies: ", aparam)
  trndata, tstdata, clusterdness = makeset(loaddataset(dataset, aparam.type ,idir)..., dataparts[1], aparam.variation, repetition)

  X = subsampleanomalous(trndata, aparam.polution)[1]
  model, modelparams = fit(X)
  BSON.@save oprefix*"_model.bson" model
  stats = merge(supervisedstats(model, trndata, "train_"),
    supervisedstats(model, tstdata, "test_"),
    unsupervisedstats(model, X, "unsup_"),
    modelparams,
    (repetition = 1, dataset = dataset)
    )
  BSON.@save oprefix*"_stats.bson" stats aparam dataparts
end

function supervisedstats(ŷ::Vector{T}, y::Vector{Int}, prefix::String = "") where {T<:Real}
  dld01, dld05 = quantile(ŷ[y .== 1], (0.01, 0.05))
  lkl = mean(ŷ[y .== 1])

  auc = EvalCurves.auc(EvalCurves.roccurve(-ŷ, y .- 1)...)
  names = tuple(map(p -> Symbol(prefix,p), ["dld01", "dld05", "lkl", "auc"])...)
  NamedTuple{names}([dld01, dld05, lkl, auc])
end
supervisedstats(model, x, y::Vector{Int}, prefix::String = "") = supervisedstats(logpdf(model, x), y, prefix)
supervisedstats(model, data::Tuple, prefix::String = "") = supervisedstats(model, data..., prefix)

function unsupervisedstats(ŷ::Vector{T}, prefix::String = "") where {T<:Real}
  dld01, dld05 = quantile(ŷ, (0.01, 0.05))
  lkl = mean(ŷ)
  names = tuple(map(p -> Symbol(prefix,p), ["dld01", "dld05", "lkl"])...)
  NamedTuple{names}([dld01, dld05, lkl])
end
unsupervisedstats(model, x, y) = unsupervisedstats(logpdf(model, x))
