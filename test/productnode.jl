using SumDenseProduct, Test, Distributions

@testset "ProductNode" begin
	p = (MvNormal([0],1), MvNormal([1],1))
	x = randn(2,10)
	m = ProductNode(p)
	@test logpdf(m, x) ≈ logpdf(p[1], x[1:1,:]) + logpdf(p[2], x[2:2,:])
	@test length(p) == 2
end
