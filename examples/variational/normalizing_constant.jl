using SumProductTransform:
	SumNode,
	TransformationNode,
	SVDDense,
	ScaleShift,
	samplebatch,
	logpdf,
	# maptree,
	sampletree,
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

function gd!(m::SumNode,
			 x::Array{Float64, 2},
			 maxsteps::Int;
			 opt = ADAM(),
			 ps = params(m))

	for t in 1:maxsteps
		gs = gradient(() -> -mean(logpdf(m, x)), ps)
		Optimise.update!(opt, ps, gs)

		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end
end

function gd!(m::SumNode,
			 x::Array{Float64, 2},
			 maxsteps::Int,
			 batchsize::Int;
			 opt = ADAM(),
			 ps = params(m),
			 normconst::Symbol = :exact)

	for t in 1:maxsteps
		batch = samplebatch(x, batchsize)
		gs = gradient(() -> logpdf(m, batch[:, 1]), ps)

		foreach(p -> gs[p] !== nothing && gs[p] .= 0, ps)
		foreach(1:batchsize) do i
            bi = batch[:, i]
			# ti = maptree(m, batch[:, i:i])[2][1]
			ti = sampletree(m)

			gi = gradient(() -> -mean(exp.(treelogpdf(m, bi, ti))), ps)
			if normconst == :exact
				ni = mean(exp.(logpdf(m, bi)))

			elseif normconst == :sample
				ni = mean(exp.(treelogpdf(m, bi, ti)))

			else
				@error "unknown setting"
			end

			foreach(p -> gs[p] .= gs[p] == nothing ? gi[p] ./ ni : gs[p] .+ gi[p] ./ ni, filter(p -> !isnothing(gi[p]), collect(ps)))
		end
		foreach(p -> gs[p] !== nothing && gs[p] ./= batchsize, ps)

		Optimise.update!(opt, ps, gs)
		mod(t, 1000) == 0 && @show mean(logpdf(m, x))
	end
end

x = flower2(200, npetals = 9)

m1 = sptn(2, 9, 1)
m2 = sptn(2, 9, 1)
m3 = sptn(2, 9, 1)

gd!(m1, x, 10000)
println("done")
gd!(m2, x, 10000, 100, normconst = :exact)
println("done")
gd!(m3, x, 10000, 100, normconst = :sample)
println("done")