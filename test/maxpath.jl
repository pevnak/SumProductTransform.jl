using SumDenseProduct, Distributions, Unitary, Test
using SumDenseProduct: mappath, pathlogpdf, batchpathlogpdf

@testset "testing the mappath" begin
	d = 4
	x = randn(d,10)
	p = MvNormal(4,1)
	m1 = DenseNode(Unitary.SVDDense(d, identity), p)
	m2 = DenseNode(Unitary.SVDDense(d, identity), p)

	@test mappath(m1, x)[1] ≈ logpdf(m1, x)
	@test all(typeof.(mappath(m1, x)[2]) .==  typeof(tuple()))

	m = SumNode([m1,m2], [0.5,0.5])
	@test mappath(m, x)[1] ≈ max.(logpdf(m1, x),logpdf(m2, x))
	@test map(s -> s[1], mappath(m, x)[2]) ≈ Flux.onecold(vcat(logpdf(m1, x)',logpdf(m2, x)'))
	m2, m1 = m1,m2
	m = SumNode([m1,m2], [0.5,0.5])
	@test map(s -> s[1], mappath(m, x)[2]) ≈ Flux.onecold(vcat(logpdf(m1, x)',logpdf(m2, x)'))

	m = ProductNode((m1, m2))
	x = randn(length(m), 10)
	mappath(m, x)[1] ≈ logpdf(m, x)

	d = 8
	x = rand(d,10)
	m = buildmixture(d, [4,4], [identity, identity], [4,0])
	lkl, path = mappath(m, x)
	lkl ≈ batchpathlogpdf(m, x, path)
end
