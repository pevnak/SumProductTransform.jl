using SumDenseProduct, Test
include("builder.jl")
include("productnode.jl")


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