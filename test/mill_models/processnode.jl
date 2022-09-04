using SumProductTransform, Test, Distributions, Flux
using Mill

@testset "ProcessNode --- forward" begin
	m = ProcessNode(MvNormal([0., 0.], [1. 0.; 0. 1.]), Poisson(2.))

    n = 10
	x = randn(2, n)
    a = rand(1:n-2, n)
    b = map(i -> rand(a[i]:n), 1:n)
    bagids = map(i -> a[i]:b[i], 1:n)
    AN = ArrayNode(x)
    BN = BagNode(AN, bagids)


	@test logpdf(m, BN) != nothing
	@test length(m) == 2
end

@testset "ProcessNode --- integration with Flux" begin

    Flux.@functor Poisson
	m = ProcessNode(MvNormal([0., 0.], [1. 0.; 0. 1.]), Poisson(2.))
	
    n = 10
	x = randn(2, n)
    a = rand(1:n-2, n)
    b = map(i -> rand(a[i]:n), 1:n)
    bagids = map(i -> a[i]:b[i], 1:n)
    AN = ArrayNode(x)
    BN = BagNode(AN, bagids)

	ps = Flux.params(m);


    # @test gradient(() -> sum(logpdf(m, BN)), ps) != nothing
	@test isempty(Flux.params(m))
end