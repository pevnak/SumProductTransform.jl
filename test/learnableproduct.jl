using SumDenseProduct, Distributions
using SumDenseProduct: FullScope, Scope, samplepath, treelogpdf, SVDDense

m = DenseNode(
	SVDDense(2, identity),
	SumNode([LearnableProductNode(2,
		DenseNode(	
				SVDDense(2, identity),
				LearnableProductNode(2, 
					Distributions.MvNormal(2,1f0))))
			 for _ in 1:4])
	)

path = samplepath(m, FullScope(2))

x = randn(2, 10)
treelogpdf(m, x, path)