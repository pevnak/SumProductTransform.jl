### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 0f141066-f9cb-11ea-1167-079c00206af2
begin	
	using Pkg
	Pkg.activate(".")
	using ToyProblems, SumProductTransform, Unitary, Flux, Setfield
	using SumProductTransform: fit!,logpdf
	using ToyProblems: flower2
	using DistributionsAD: TuringMvNormal
	using SumProductTransform: ScaleShift, SVDDense
	using Plots
	plotly()
end;

# ╔═╡ c526fbdc-fa4f-11ea-34fa-075192c6fce4
md"""# A gentle introduction to SumProductTransform library

This introduction uses several unregistered libraries, namely `ToyProblems.jl`,  `SumProductTransform.jl` which depends on `Unitary.jl`. The best is to `instantiate` environment in `example/` directory, which should continue all you need including Pluto.

The intruduction starts with a classic **Gaussian Mixture Model**, continues with a simple **Sum Product Network** and graduates with **Sum Product Transform Network**.

Before we dive into real business, we import libraries and define a convenient function for plotting densities and data.
"""

# ╔═╡ c6400e6e-fa54-11ea-30d1-2d0c931aa9d6
md"""A plotting function will show the density of a fitted model and that of with training data on top""" 

# ╔═╡ 11c6593a-fa51-11ea-0335-ebb129ca619b
function plot_contour(m, x, title = nothing)
	levels = quantile(exp.(logpdf(m, x)), 0.01:0.09:0.99)
	δ = levels[1] / 10
	levels = vcat(collect(levels[1] - 10δ:δ:levels[1] - δ), levels)
	xr = range(minimum(x[1,:]) - 1 , maximum(x[1,:])+ 1 , length = 200)
	yr = range(minimum(x[2,:]) - 1 , maximum(x[2,:])+ 1 , length = 200)
	p1 = Plots.contour(xr, yr, (x...) ->  exp(logpdf(m, [x[1],x[2]])[1]))
	p2 = deepcopy(p1)
	xx = x[:,sample(1:size(x,2), 100, replace = false)]
	scatter!(p2, x[1,:], x[2,:], alpha = 0.4)
	p = plot(p1, p2)
	p
end;

# ╔═╡ cdc55a98-fa3d-11ea-33a5-19b235af1534
md""" Let's create training samples from **Flower** dataset with nine petals."""

# ╔═╡ 0c98beb6-fa36-11ea-1d28-61ff40782858
x = flower2(999, npetals = 9);

# ╔═╡ 02631b66-fa3e-11ea-31e2-15c6b240812f
md""" Initialize dimension of data `d`, batchsize in stochastic gradient descend, and number of training steps"""

# ╔═╡ ae3549ce-f9cb-11ea-2240-2f967be74b0f
begin
	d = size(x,1)
	batchsize = 100
	nsteps = 20000
end;

# ╔═╡ 5f549206-fa3e-11ea-06c6-a7cb7ce38cfc
md"""### Gaussian Mixture Model

`gmm` with 144 components"""

# ╔═╡ b89ca116-f9cb-11ea-01a3-3fa73e7d6645
begin
	ngmm_components = 144
	init_normal(d) = TransformationNode(SVDDense(d), TuringMvNormal(d, 1f0))
	gmm_components = [init_normal(d) for i in 1:ngmm_components]
	gmm = SumNode(gmm_components)
end;

# ╔═╡ c4e1ea62-f9cb-11ea-2aea-b3391fcb8b04
fit!(gmm, x, batchsize, nsteps);

# ╔═╡ cdc3cc9a-f9cb-11ea-2c50-9bf39b9cf21d
plot_contour(gmm, x)

# ╔═╡ 62a58dc0-fa54-11ea-3797-6f179aabbc3d
md"""### Sum Product network"""

# ╔═╡ d66819e6-fa39-11ea-3cba-a3e5e4df656a
begin 
	spt_ncomponents = 9
	Normal1D() = TransformationNode(ScaleShift(1), TuringMvNormal(1, 1f0));
	spn_components = map(1:spt_ncomponents) do _
			p₁ = SumNode([Normal1D() for _ in 1:spt_ncomponents])
			p₂ = SumNode([Normal1D() for _ in 1:spt_ncomponents])
			p₁₂ = ProductNode((p₁, p₂))
		end
	spn = SumNode(spn_components)
end;

# ╔═╡ 109c0ad2-fa3a-11ea-06f8-47af0faf1913
fit!(spn, x, batchsize, nsteps);

# ╔═╡ 1877db00-fa3a-11ea-0054-512ce17a6f8f
plot_contour(spn, x)

# ╔═╡ 48e61ed6-fa54-11ea-0abb-c940913bd662
md"""### Sum Product Transform network
with affine transformations and Normal distribution on leaves"""

# ╔═╡ 3c6eb202-fa3c-11ea-38d3-610ae96a3755
begin 
	nsptn_components = 3
	global sptn = TransformationNode(ScaleShift(2), TuringMvNormal(2, 1f0));
	for i in 1:3
		global sptn
		sptn = SumNode([TransformationNode(SVDDense(2), sptn) for i in 1:nsptn_components])
	end
end;

# ╔═╡ 9e010d38-fa3c-11ea-39d7-bb0e0d22cd38
fit!(sptn, x, batchsize, nsteps);

# ╔═╡ a6291cb4-fa3c-11ea-1364-d3a4ef782b83
plot_contour(sptn, x)

# ╔═╡ 75b4eac0-f9cb-11ea-0ac4-718f4827c464
md"""### Sum Product Transform network 

with nonlinear transformation on leaves"""

# ╔═╡ 5e8c722c-fa53-11ea-2c10-b57c7f9119c1
begin 
	leaf = ProductNode((
		SumNode([TransformationNode(Chain(SVDDense(1, selu), ScaleShift(1)), TuringMvNormal(1, 1f0))  for _ in 1:3]),
		SumNode([TransformationNode(Chain(SVDDense(1, selu), ScaleShift(1)), TuringMvNormal(1, 1f0))  for _ in 1:3]),
			))
			
	global sptn2 = leaf
	for i in 1:3
		global sptn2
		sptn2 = SumNode([TransformationNode(SVDDense(2), sptn2) for i in 1:3])
	end
end;

# ╔═╡ 21edf312-fa54-11ea-2206-d7e183b883fb
fit!(sptn2, x, batchsize, nsteps);

# ╔═╡ 43b82e04-fa54-11ea-0d41-8d773b048a9b
plot_contour(sptn2, x)

# ╔═╡ Cell order:
# ╟─c526fbdc-fa4f-11ea-34fa-075192c6fce4
# ╠═0f141066-f9cb-11ea-1167-079c00206af2
# ╟─c6400e6e-fa54-11ea-30d1-2d0c931aa9d6
# ╠═11c6593a-fa51-11ea-0335-ebb129ca619b
# ╟─cdc55a98-fa3d-11ea-33a5-19b235af1534
# ╟─0c98beb6-fa36-11ea-1d28-61ff40782858
# ╟─02631b66-fa3e-11ea-31e2-15c6b240812f
# ╠═ae3549ce-f9cb-11ea-2240-2f967be74b0f
# ╟─5f549206-fa3e-11ea-06c6-a7cb7ce38cfc
# ╠═b89ca116-f9cb-11ea-01a3-3fa73e7d6645
# ╠═c4e1ea62-f9cb-11ea-2aea-b3391fcb8b04
# ╠═cdc3cc9a-f9cb-11ea-2c50-9bf39b9cf21d
# ╠═62a58dc0-fa54-11ea-3797-6f179aabbc3d
# ╠═d66819e6-fa39-11ea-3cba-a3e5e4df656a
# ╠═109c0ad2-fa3a-11ea-06f8-47af0faf1913
# ╠═1877db00-fa3a-11ea-0054-512ce17a6f8f
# ╠═48e61ed6-fa54-11ea-0abb-c940913bd662
# ╠═3c6eb202-fa3c-11ea-38d3-610ae96a3755
# ╠═9e010d38-fa3c-11ea-39d7-bb0e0d22cd38
# ╠═a6291cb4-fa3c-11ea-1364-d3a4ef782b83
# ╟─75b4eac0-f9cb-11ea-0ac4-718f4827c464
# ╠═5e8c722c-fa53-11ea-2c10-b57c7f9119c1
# ╠═21edf312-fa54-11ea-2206-d7e183b883fb
# ╠═43b82e04-fa54-11ea-0d41-8d773b048a9b
