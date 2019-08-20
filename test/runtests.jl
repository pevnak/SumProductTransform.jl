using SumDenseProduct, Test, Distributions
include("builder.jl")
include("productnode.jl")
include("maxpath.jl")


@testset "sampling" begin
	ns = [2,2]
	σs = [identity, identity]
	noise = [4,0]
	d = 8
	for s in [:all, :none, :dense]
	    m = buildmixture(d, ns, σs, noise, sharing = s)
	    @test length(rand(m)) == 8
	end
end

@testset "batch calculation of likelihood" begin 
	p = MvNormal(3, 1)
	x = randn(3,100)
	@test logpdf(p, x) ≈ batchlogpdf(p, x, 32)
	ns = [2,2]
	σs = [identity, identity]
	noise = [4,0]
	d = 8
	for s in [:all, :none, :dense]
	    m = buildmixture(d, ns, σs, noise, sharing = s)
	    x = randn(length(m), 100)
	    @test logpdf(m, x) ≈ batchlogpdf(m, x, 32)
	end	
end
