using DrWatson
@quickactivate
using SumProductTransform: fit!, mhsaem!, buildmixture
# using ADatasets: makeset, loaddataset
using ToyProblems: flower2

function main()
	n = 128
    l = 2
    batchsize = 100
    steps = 20
    samples = 1

    dataset = "sonar"
    dir = "/home/pevnytom/Data/numerical"

    x = flower2(200, npetals=9)
    # x, _, _ = makeset(loaddataset(dataset, "easy", dir)..., 0.8, "low", 1)
    m = buildmixture(size(x, 1), n, l, identity; sharing=:none)

    fit!(m, x, batchsize, steps, check=1)
    mhsaem!(m, x, batchsize, steps, samples, check=1)
end

main()

# include("examples/variational/sgd_vs_mhsaem.jl")
