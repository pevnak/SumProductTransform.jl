using SumProductTransform, Test, LinearAlgebra, Flux
using SumProductTransform: SVDDense, ScaleShift, forward, logabsdetjac
using ForwardDiff

@testset "Can I invert SVDDense and its chain" begin
	for d in [1, 2, 3, 4]
		for m in [SVDDense(d), SVDDense(d, selu), Chain(SVDDense(d, identity), SVDDense(d, identity)), Chain(SVDDense(d, selu), SVDDense(d, selu))]
			mi = inv(m)
			for x in [rand(d), rand(d,10), transpose(rand(10, d))]
				@test isapprox(mi(m(x)),  x, atol = 1e-3)
			end
		end
	end

	for d in [1, 2, 3, 4]
		for m in [ScaleShift(d, identity), ScaleShift(d, selu), Chain(ScaleShift(d, identity), ScaleShift(d, identity)), Chain(ScaleShift(d, selu), ScaleShift(d, selu))]
			mi = inv(m)
			for x in [rand(d), rand(d,10), transpose(rand(10, d))]
				@test isapprox(mi(m(x)),  x, atol = 1e-3)
			end
		end
	end
end

@testset "testing the determinant" begin
	x = randn(2)
	for m in [SVDDense(2, identity), SVDDense(2, selu), SVDDense(2, leakyrelu), SVDDense(2, tanh)]
		y, ladj = forward(m, x)
		@test m(x) ≈ y 
		@test inv(m)(y) ≈ x rtol = 0.0001
	    @test log(abs(det(ForwardDiff.jacobian(m, x)))) ≈ sum(ladj) rtol = 1e-4
	    @test log(abs(det(ForwardDiff.jacobian(inv(m), y)))) ≈ sum(logabsdetjac(inv(m), y)) rtol = 1e-4
	end

	for m in [ScaleShift(2, identity), ScaleShift(2, selu), ScaleShift(2, leakyrelu), ScaleShift(2, tanh)]
		y, ladj = forward(m, x)
		@test inv(m)(y) ≈ x rtol = 0.0001
		@test m(x) ≈ y 
	    @test log(abs(det(ForwardDiff.jacobian(m, x)))) ≈ sum(ladj) rtol = 1e-4
	    @test log(abs(det(ForwardDiff.jacobian(inv(m), y)))) ≈ sum(logabsdetjac(inv(m), y)) rtol = 1e-4
	end
end
