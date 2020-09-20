using SumProductTransform, Test, Distributions
# include("builder.jl")
include("layers/diagonalrectangular.jl")
include("layers/svd.jl")
include("layers/inverse.jl")
include("layers/ludense.jl")
include("layers/jacobian.jl")

include("productnode.jl")
include("maxtree.jl")


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

@testset "segmented calculation of gradient" begin 
	ns = [2,2]
	σs = [identity, identity]
	noise = [4,0]
	d = 8
	for s in [:all, :none, :dense]
	    m = buildmixture(d, ns, σs, noise, sharing = s)
	    x = randn(length(m), 100)
	    segments = [i:i+9 for i in 1:10:100]
	    ps = Flux.params(m)
	    gr1 = SumProductTransform.threadedgrad(i -> sum(logpdf(m, x[:,i])), ps, segments)
	    gr2 = gradient(() -> mean(logpdf(m, x)), ps)
	    @test all(isapprox(gr1[p] ./ 100, gr2[p], atol = 1e-6) for p in ps)
	    # for p in ps
     #       if !(gr1[p] ./ 100 ≈ gr2[p])
     #       	@show gr1[p]
     #       	@show gr2[p]
     #       end
     #   end
	end	
end
