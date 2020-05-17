using ToyProblems, Distributions, SumDenseProduct, Unitary, Flux, Setfield
using Flux:throttle
using SumDenseProduct: fit!, maptree, sampletree
using ToyProblems: flower2

using Plots
plotly()

function plot_contour(m, x)
	levels = quantile(exp.(logpdf(m, x)), 0.01:0.09:0.99)
	xr = range(minimum(x[1,:]) - 1 , maximum(x[1,:])+ 1 , length = 200)
	yr = range(minimum(x[2,:]) - 1 , maximum(x[2,:])+ 1 , length = 200)
	# contour(xr, yr, (x...) ->  logpdf(m, [x[1],x[2]])[1], levels = levels)
	# heatmap(xr, yr, (x...) ->  exp(logpdf(m, [x[1],x[2]])[1]))
	# scatter!(x[1,:], x[2,:])
	contour(xr, yr, (x...) ->  exp(logpdf(m, [x[1],x[2]])[1]))
end

function plot_components(m, x)
	tree = hash.(maptree(m, x)[2])
	u = unique(tree)
	hash2int = Dict(u[i] => i for i in 1:length(u))
	i = [hash2int[k] for k in tree]
	scatter(x[1,:], x[2,:], color = i)
end

function plot_rand(m, n)
	xx = reduce(hcat, rand(m) for i in 1:n)
	scatter(xx[1,:], xx[2,:])
end

function buildmbf(n, decom = :svdgivens)
	p₁₂ = SumNode([TransformNode(Chain(Unitary.Transform(2, selu, :svdgivens), Unitary.Transform(2, identity, :svdgivens)),  MvNormal(2,1f0)) for _ in 1:2])
	SumNode([TransformNode(Unitary.Transform(2, identity, :svdgivens), p₁₂) for i in 1:n])
end

function buildm2(decom = :svdgivens)
	p₁₂ = TransformNode(Chain(Unitary.Transform(2, selu, decom), Unitary.Transform(2, identity, decom)),  MvNormal(2,1f0))
	m₁ = SumNode([TransformNode(Chain(Unitary.Transform(2, identity, decom),Unitary.Transform(2, identity, decom)), p₁₂) for i in 1:9])
	SumNode([TransformNode(Chain(Unitary.Transform(2, identity, decom), Unitary.Transform(2, identity, decom)), m₁) for i in 1:9])
end

function buildm3(decom=:svdgivens)
	p₁₂ = TransformNode(Chain(Unitary.Transform(2, selu, decom), Unitary.Transform(2, identity, decom)),  MvNormal(2,1f0))
	m₁ = SumNode([TransformNode(Chain(Unitary.Transform(2, identity, decom),Unitary.Transform(2, identity, decom)), p₁₂) for i in 1:6])
	m2 = SumNode([TransformNode(Chain(Unitary.Transform(2, identity, decom), Unitary.Transform(2, identity, decom)), m₁) for i in 1:6])
	SumNode([TransformNode(Chain(Unitary.Transform(2, identity, decom), Unitary.Transform(2, identity, decom)), m2) for i in 1:6])
end


###############################################################################
#			non-normal mixtures
###############################################################################
x = flower2(1000, npetals = 9)
#model = buildmbf(9)
model = buildmlu(9)
#SumDenseProduct.initpp!(model, x, 9)
history = fit!(model, x, 64, 10000, 100; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())
plot_contour(model, x);
title!("non-normal mixture")
plot_components(model, x)

# init_model = buildm2()
# SumDenseProduct.initpp2!(model, x, 20)
# updatelatent!(model, x)
# plot_contour(model, x);
# title!("initilized")
# plot_components(model, x)
model = buildm2()
history = fit!(model, x, 64, 10000, 20; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())
plot_contour(model, x);
title!("fitted")
plot_components(model, x)
plot_rand(model[1].c.p[1].c)

# Let's try to find a sample with least likelihood

function iteratedlearning(model, x, reset_threshold = 0.01)
	history = fit!(model, x, 64, 5000, 20; gradmethod = :sampling, minimum_improvement = -1e10, opt = ADAM())
	for i in 1:10 
		@show mean(logpdf(model, x))
		os, trees = maptree(model, x)
		i = argmin(os)
		tree = trees[i]
		xx = x[:,i:i]
		if minimum(model.prior) < reset_threshold
			ci = argmin(model.prior)
			newtree = @set tree[1] = ci
			ps = Flux.params(model[ci].c.m)
			SumDenseProduct.inittree!(model, xx, newtree, ps , verbose = true)
		end
		if minimum(model[1].c.p.prior) < reset_threshold
			ci = argmin(model[1].c.p.prior)
			newtree = @set tree[2][1] = ci
			ps = Flux.params(model[1].c.p[ci].c.m)
			SumDenseProduct.inittree!(model, xx, newtree, ps , verbose = true)
		end
		updatelatent!(model, x)
		@show mean(logpdf(model, x))
		history = fit!(model, x, 64, 5000, 20; gradmethod = :sampling, minimum_improvement = -1e10, opt = ADAM())
		println("fit finished")
	end
end
model = buildm2()
iteratedlearning(model, x)
plot_contour(model, x);
title!("fitted")
plot_components(model, x)


###############################################################################
#			non-normal mixtures
###############################################################################
model = buildm2(18)
history = fit!(model, x, 64, 20000, 100; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())
plot_contour(model, x);
title!("non-normal mixture2")

###############################################################################
#			normal mixture
###############################################################################
model = buildmixture(2, 18, 1, identity; sharing = :transform, firsttransform = false)
history = fit!(model, x, 64, 20000, 100; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())
plot_contour(model, x);
title!("normal mixture")

###############################################################################
#			Let's try shared inner mixture
###############################################################################
m = allsharedmixture(2, 4, 2, identity)
ps = Flux.params(m);
opt = ADAM()
Flux.train!(i -> -mean(logpdf(m, x)), Flux.Params(ps), 1:5000, opt; cb = throttle(() -> (@show mean(logpdf(m, x))),10))
updatelatent!(m, x);
plot_contour(m, x)

###############################################################################
#			Let's try shared more
###############################################################################
mmi = createmixture(3);
mi = createmixture(3, identity, () -> mmi);
m = createmixture(3, identity,() -> mi);
ps = Flux.params(m);
opt = RMSProp(0.01)
Flux.train!(i -> -mean(logpdf(m, x)), Flux.Params(ps), 1:5000, opt; cb = throttle(() -> (@show mean(logpdf(m, x))),10))
updatelatent!(m, x); 
plot_contour(m, x)
