using SumProductTransform, Test, Distributions, Flux
using SumProductTransform: PoissonA
using Mill

@testset "PoissonA" begin
	m = PoissonA(5)
    xs = [rand(1:20, 10), 2, []]

    for x in xs
        @test size(logpdf(m, x)) == size(x)
    end
end

@testset "PoissonA --- integration with Flux" begin

    Flux.@functor PoissonA
	m = PoissonA(5)
    truegrad(λ, x) = -1 + x/λ         

	ps = Flux.params(m);

    @test !isempty(ps)
    @test gradient(() -> sum(logpdf(m, rand(1:20, 10))), ps) != nothing
    x = 10
    @test gradient(() -> logpdf(m, x), ps).grads[ps[1]] ≈ truegrad(m.λ[], x)
end