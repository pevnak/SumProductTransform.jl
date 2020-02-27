using SumDenseProduct, Test
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

Unitary.Butterfly(a::Butterfly, s::Scope) = 
	Butterfly(filter(s,a.θs, a.idxs)..., length(s.dims))
Unitary.DiagonalRectangular(a::DiagonalRectangular, s::Scope) = 
	DiagonalRectangular(a.d[s.dims], length(s.dims), length(s.dims))
Unitary.SVDDense(a::SVDDense, s::Scope) = 
	SVDDense(Butterfly(a.u, s), DiagonalRectangular(a.d, s), Butterfly(a.v, s), a.b[s.dims], a.σ)

@testset "scoping in SVDDense" begin
    m  = Unitary.SVDDense(4, identity, :givens)
    x = Float32.(reshape([1,2,3,4],4,1))
    xx, _= m((x, 0))

    s = Scope([1,2],4)
    ms = SVDDense(m, s)
    xx, l, _= m((x[1:2,:], 0, s))
	xxr, lr = ms((x[1:2,:],0))
	@test xx ≈ xxr
	@test l ≈ lr

    s = Scope([1,2,3],4)
    ms = SVDDense(m, s)
    xx, l, _= m((x[1:3,:], 0, s))
	xxr, lr = ms((x[1:3,:],0))
	@test xx ≈ xxr
	@test l ≈ lr

    s = Scope([1,3,4],4)
    ms = SVDDense(m, s)
    xx, l, _= m((x[1:3,:], 0, s))
	xxr, lr = ms((x[1:3,:],0))
	@test xx ≈ xxr
	@test l ≈ lr
end

@testset "scoping and Flux" begin
    m  = Unitary.SVDDense(4, identity, :givens)
    x = Float32.(reshape([1,2,3,4],4,1))
    xx, _= m((x, 0))

    s = Scope([1,2],4)
    ms = SVDDense(m, s)
    xx, l, _= m((x[1:2,:], 0, s))
	xxr, lr = ms((x[1:2,:],0))
	@test xx ≈ xxr
	@test l ≈ lr

    s = Scope([1,2,3],4)
    ms = SVDDense(m, s)
    xx, l, _= m((x[1:3,:], 0, s))
	xxr, lr = ms((x[1:3,:],0))
	@test xx ≈ xxr
	@test l ≈ lr

    s = Scope([1,3,4],4)
    ms = SVDDense(m, s)
    xx, l, _= m((x[1:3,:], 0, s))
	xxr, lr = ms((x[1:3,:],0))
	@test xx ≈ xxr
	@test l ≈ lr
end

