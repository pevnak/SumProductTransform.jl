using SumDenseProduct, Distributions, Unitary, Test
using SumDenseProduct: maptree, treelogpdf, batchtreelogpdf

@testset "testing the maptree" begin
	d = 4
	x = randn(d,10)
	p = MvNormal(4,1)
	m1 = TransformationNode(Unitary.SVDDense(d, identity), p)
	m2 = TransformationNode(Unitary.SVDDense(d, identity), p)

	@test maptree(m1, x)[1] ≈ logpdf(m1, x)
	@test all(typeof.(maptree(m1, x)[2]) .==  typeof(tuple()))

	m = SumNode([m1,m2], [0.5,0.5])
	@test maptree(m, x)[1] ≈ max.(logpdf(m1, x),logpdf(m2, x))
	@test map(s -> s[1], maptree(m, x)[2]) ≈ Flux.onecold(vcat(logpdf(m1, x)',logpdf(m2, x)'))
	m2, m1 = m1,m2
	m = SumNode([m1,m2], [0.5,0.5])
	@test map(s -> s[1], maptree(m, x)[2]) ≈ Flux.onecold(vcat(logpdf(m1, x)',logpdf(m2, x)'))

	m = ProductNode((m1, m2))
	x = randn(length(m), 10)
	maptree(m, x)[1] ≈ logpdf(m, x)

	d = 8
	x = rand(d,10)
	m = buildmixture(d, [4,4], [identity, identity], [4,0])
	lkl, path = maptree(m, x)
	lkl ≈ batchtreelogpdf(m, x, path)
end
