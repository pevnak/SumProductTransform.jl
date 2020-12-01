using SumProductTransform:
	SumNode,
	TransformationNode,
	SVDDense,
	ScaleShift,
	samplebatch,
	logpdf,
	maptree,
	treelogpdf
using DistributionsAD: TuringMvNormal
using ToyProblems: flower2
using Statistics: mean
using Flux:
	ADAM,
	Optimise,
	params,
	gradient
using StatsBase: sample


function sptn(d, n, l)
    m = TransformationNode(ScaleShift(d), TuringMvNormal(d,1f0))
    for i in 1:l
        m = SumNode([TransformationNode(SVDDense(2, identity, :butterfly), m)
            for i in 1:n])
    end
    return(m)
end

function gd!(m, x, maxsteps::Int, batchsize::Int; opt = ADAM(), ps = params(m))
	for t in 1:maxsteps
		batch = samplebatch(x, batchsize)
		gs = gradient(() -> logpdf(m, batch[:,1]), ps)
		foreach(p -> gs[p] !== nothing && gs[p] .= 0, ps)
		foreach(1:batchsize) do i
            bi = batch[:, i]
			tri = maptree(m, batch[:, i:i])[2][1]
			onegs = gradient(() -> -mean(exp.(treelogpdf(m, bi, tri))), ps) 
			val = mean(exp.(logpdf(m, bi)))
			foreach(p -> gs[p] .= gs[p] == nothing ? onegs[p] ./ val : gs[p] .+ onegs[p] ./ val, filter(p -> !isnothing(onegs[p]), collect(ps)))
		end
		foreach(p -> gs[p] !== nothing && gs[p] ./= batchsize, ps)
		Optimise.update!(opt, ps, gs)
		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end
end

x = flower2(1000, npetals = 9)
m = sptn(2, 9, 1)

gd!(m, x, 10000, 100)