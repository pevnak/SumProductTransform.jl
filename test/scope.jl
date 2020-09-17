using SumDenseProduct, Test, Flux
using Unitary
using Unitary: Butterfly, SVDDense, DiagonalRectangular
using SumDenseProduct: Scope, FullScope, NoScope

@testset "subsetting scopes" begin
	s = FullScope(4)
	r = s[1:2]

	@test r.dims == [1,2]
	@test r.n == 4
	@test r.active == [true, true, false, false]
	@test r.idxmap == [1, 2, 0, 0]

	r = s[[2,4]]
	@test r.dims == [2,4]
	@test r.n == 4
	@test r.active == [false, true, false, true]
	@test r.idxmap == [0, 1, 0, 2]

	r = s[[4,2]]
	@test r.dims == [4, 2]
	@test r.n == 4
	@test r.active == [false, true, false, true]
	@test r.idxmap == [0, 1, 0, 2]

	s = FullScope(6)
	r = s[2:4][[1,2]]
	@test r.dims == [2,3]
	@test r.n == 6
	@test r.active == [false, true, true, false, false, false]
	@test r.idxmap == [0, 1, 2, 0, 0, 0]
end

@testset "Filtering indexes" begin 
	idxs = [(j, i) for i in 1:4 for j in 1:(i-1)]
	θs = collect(1:length(idxs))

	s = Scope([1,2],4)
	rθs, ridxs = filter(s, θs, idxs)
	@test rθs == [1]
	@test ridxs == [(1,2)]

	s = Scope([1,3],4)
	rθs, ridxs = filter(s, θs, idxs)
	@test rθs == [2]
	@test ridxs == [(1,2)]


	s = Scope([3,4],4)
	rθs, ridxs = filter(s, θs, idxs)
	@test rθs == [6]
	@test ridxs == [(1,2)]

	s = Scope([1,3,4],4)
	rθs, ridxs = filter(s, θs, idxs)
	@test rθs == [2,4,6]
	@test ridxs == [(1,2),(1,3), (2,3)]

	s = Scope(4)
	rθs, ridxs = filter(s, θs, idxs)
	@test rθs == θs
	@test ridxs == idxs

	s = NoScope()
	rθs, ridxs = filter(s, θs, idxs)
	@test rθs == θs
	@test ridxs == idxs
end

@testset "scoping in SVDDense" begin
    m  = Unitary.SVDDense(4, identity, :givens)
    x = Float32.(reshape([1,2,3,4],4,1))
    xx, _= m((x, 0))

    s = Scope([1,2],4)
    ms = m[s]
    xx, l, _= m((x[1:2,:], 0, s))
	xxr, lr = ms((x[1:2,:],0))
	@test xx ≈ xxr
	@test l ≈ lr

    s = Scope([1,2,3],4)
    ms = m[s]
    xx, l, _= m((x[1:3,:], 0, s))
	xxr, lr = ms((x[1:3,:],0))
	@test xx ≈ xxr
	@test l ≈ lr

    s = Scope([1,3,4],4)
    ms = m[s]
    xx, l, _= m((x[1:3,:], 0, s))
	xxr, lr = ms((x[1:3,:],0))
	@test xx ≈ xxr
	@test l ≈ lr
end

@testset "scoping and Flux" begin
    m  = Unitary.SVDDense(4, identity, :givens)
    x = Float32.(reshape([1,2,3,4],4,1))

    ps = Flux.params(m)

    gradient(() -> sum(m((x,0))[1]), ps)


    s = Scope([1,2],4)
    gs1 = gradient(() -> sum(sin.(m((x[1:2,:],0,s))[1])), ps)
    gs2 = gradient(() -> sum(sin.(m[s]((x[1:2,:],0))[1])), ps)
	@test all([gs1[p] ≈ gs2[p] for p in ps])
end


#############################################################################
#	Benchmarking
#############################################################################
using BenchmarkTools
d, l = 50, 100
m  = Unitary.SVDDense(d, identity, :givens)
x = randn(Float32, 50, 100)

ps = Flux.params(m)

@btime gradient(() -> sum(m((x,0))[1]), ps); #  1.901 ms (269 allocations: 345.98 KiB)


s = Scope(d)[1:25]
xx = x[1:25,:]
@btime gradient(() -> sum(sin.(m((xx,0,s))[1])), ps); # 1.392 ms (67950 allocations: 2.69 MiB)
@btime gradient(() -> sum(sin.(m[s]((xx,0))[1])), ps); # 1.452 ms (68098 allocations: 2.69 MiB)

