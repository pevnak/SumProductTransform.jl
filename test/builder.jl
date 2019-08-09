using SumDenseProduct, Test, Flux, Distributions

@testset "testing model generators: number of priors and parameters" begin
	m = allsharedmixture(2, 8, 1)
	@test length(Flux.params(m)) == 32
	@test length(priors(m)) == 1

	for m in [allsharedmixture(2, 4, 2), buildmixture(2, 4, 2;sharing = :all)]
		@test length(Flux.params(m)) == 32
		@test length(priors(m)) == 2
	end

	for m in [nosharedmixture(2, 4, 2), buildmixture(2, 4, 2;sharing = :none)]
		@test length(Flux.params(m)) == 4*4 + 4*4*4
		@test length(priors(m)) == 5
	end

	for m in [densesharedmixture(2, 4, 2), buildmixture(2, 4, 2;sharing = :dense)]
		@test length(Flux.params(m)) == 32
		@test length(priors(m)) == 5
	end
end


@testset "testing model generators: noise components" begin
	for k in 1:3
		d, n, σs, noise = 4, [2,2], [identity, identity], [k,0]
		for m in [nosharedmixture(d, n, σs , noise), allsharedmixture(d, n, σs , noise), densesharedmixture(d, n, σs , noise)]
			@test length(m) == d
			if k > 0
				@test length(m.components[1].p[2]) == d - k
			else
				@test length(m.components[1].p) == d
			end
			@test logpdf(m, randn(d,10)) != nothing
		end
	end
end


@testset "testing model generators: different number of components in layers" begin
	for (ks, σs)  in [([2,3,4], [identity, selu, tanh]), ([4,3,2], [tanh, selu, identity])]
		for m in [allsharedmixture(2, ks, σs), buildmixture(2, ks, σs; sharing = :all)]
			@test length(m.components) == ks[1]
			@test m.components[1].m.σ == σs[1]
			@test length(m.components[1].p.components) == ks[2]
			@test m.components[1].p.components[1].m.σ == σs[2]
			@test length(m.components[1].p.components[1].p.components) == ks[3]
			@test m.components[1].p.components[1].p.components[1].m.σ == σs[3]
		end

		for m in [nosharedmixture(2, ks, σs), buildmixture(2, ks, σs; sharing = :none)]
			@test length(m.components) == ks[1]
			@test m.components[1].m.σ == σs[1]
			@test length(m.components[1].p.components) == ks[2]
			@test m.components[1].p.components[1].m.σ == σs[2]
			@test length(m.components[1].p.components[1].p.components) == ks[3]
			@test m.components[1].p.components[1].p.components[1].m.σ == σs[3]
		end

		for m in [densesharedmixture(2, ks, σs), buildmixture(2, ks, σs; sharing = :dense)]
			@test length(m.components) == ks[1]
			@test m.components[1].m.σ == σs[1]
			@test length(m.components[1].p.components) == ks[2]
			@test m.components[1].p.components[1].m.σ == σs[2]
			@test length(m.components[1].p.components[1].p.components) == ks[3]
			@test m.components[1].p.components[1].p.components[1].m.σ == σs[3]
		end
	end
end

