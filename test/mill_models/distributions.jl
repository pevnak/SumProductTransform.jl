using SumProductTransform, Test, Distributions, Flux

@testset "PoissonA --- logpdf forward" begin
	m = PoissonA(5)
    xs = [rand(1:20, 10), 2, []]

    for x in xs
        @test size(logpdf(m, x)) == size(x)
    end

    @test logpdf(PoissonA([10, 5]), xs[1]) ≈ logpdf(PoissonA(10), xs[1]) + logpdf(PoissonA(5), xs[1])

end

@testset "PoissonA --- rand sampling" begin
    m = PoissonA(6)
    @test length(rand(m)) == length(m.λ)

    m = PoissonA([2, 7, 10])
    @test length(rand(m)) == length(m.λ)
end

@testset "PoissonA --- integration with Flux" begin

	m = PoissonA(5)
    truegrad(λ, x) = -1 .+ x./λ
	ps = Flux.params(m);

    @test !isempty(ps)
    @test gradient(() -> sum(logpdf(m, rand(1:20, 10))), ps) != nothing
    x = 10
    @test gradient(() -> logpdf(m, x), ps).grads[ps[1]] ≈ truegrad(m.λ, x)
end