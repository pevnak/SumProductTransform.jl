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

	for m in [transformsharedmixture(2, 4, 2), buildmixture(2, 4, 2;sharing = :transform)]
		@test length(Flux.params(m)) == 32
		@test length(priors(m)) == 5
	end
end


@testset "testing model generators: noise components" begin
	n, σs = [2,2], [identity, identity]
	for k in 1:3
		d = 3*k
		for noise in [[k,k],[k,0]]
			for m in [nosharedmixture(d, n, σs , noise), allsharedmixture(d, n, σs , noise), transformsharedmixture(d, n, σs , noise)]
				@test length(m) == d
				@test length(m.components[1].p) == d
				@test length(m.components[1].p[1]) == k
				@test length(m.components[1].p[2]) == d - k
				@test logpdf(m, randn(d,10)) != nothing
			end
		end
	end
end

@testset "number of tree in a model" begin
	noise = [0.5, 0.25, 0]
	d = 10
	for nc in [[2,2,2], [1,2,3], [3, 2, 1],]
		model = buildmixture(d, nc,[identity, identity, identity], round.(Int,noise.*d); sharing = :all)
		@test treecount(model) == prod(nc)
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

		for m in [transformsharedmixture(2, ks, σs), buildmixture(2, ks, σs; sharing = :transform)]
			@test length(m.components) == ks[1]
			@test m.components[1].m.σ == σs[1]
			@test length(m.components[1].p.components) == ks[2]
			@test m.components[1].p.components[1].m.σ == σs[2]
			@test length(m.components[1].p.components[1].p.components) == ks[3]
			@test m.components[1].p.components[1].p.components[1].m.σ == σs[3]
		end
	end
end

@testset "testing that a correct type of unitary matrix is constructed" begin
	for s in [:none, :all, :transform]
		model = buildmixture(2, 2, 2, identity; sharing = s, firsttransform = false, unitary = :givens)
		@test typeof(model[1].c.m.u) <: Unitary.InPlaceUnitaryUnitaryGivens
		@test typeof(model[1].c.p[1].c.m.u) <: Unitary.InPlaceUnitaryUnitaryGivens

		model = buildmixture(2, 2, 2, identity; sharing = s, firsttransform = false, unitary = :householder)
		@test typeof(model[1].c.m.u) <: Unitary.UnitaryHouseholder
		@test typeof(model[1].c.p[1].c.m.u) <: Unitary.UnitaryHouseholder

		model = buildmixture(2, 2, 2, identity; sharing = s, firsttransform = true, unitary = :givens)
		@test typeof(model.p[1].c.m.u) <: Unitary.InPlaceUnitaryUnitaryGivens
		@test typeof(model.p[1].c.p[1].c.m.u) <: Unitary.InPlaceUnitaryUnitaryGivens

		model = buildmixture(2, 2, 2, identity; sharing = s, firsttransform = true, unitary = :householder)
		@test typeof(model.p[1].c.m.u) <: Unitary.UnitaryHouseholder
		@test typeof(model.p[1].c.p[1].c.m.u) <: Unitary.UnitaryHouseholder
	end
end

