using SumDenseProduct, Test, Distributions, Flux

@testset "ProductNode --- forward" begin
	p = (MvNormal([0],1), MvNormal([1],1))
	x = randn(2,10)
	m = ProductNode(p)
	@test logpdf(m, x) ≈ logpdf(p[1], x[1:1,:]) + logpdf(p[2], x[2:2,:])
	@test length(p) == 2
end

@testset "ProductNode --- integration with Flux" begin
	p = (MvNormal([0],1), MvNormal([1],1))
	x = randn(2,10)
	m = ProductNode(p)
	ps = Flux.params(m);
	@test gradient(() -> sum(logpdf(m, x)), ps) != nothing
	@test isempty(Flux.params(m))
end
