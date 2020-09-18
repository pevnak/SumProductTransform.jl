# SumProductTransform.jl

An experimental implementation of a generalization of a Sum-Product networks by a Dense node.

*Background:* The Sum-Product networks is a hierarchical model with a tree structure composed by a two type of nodes and lists being probability distributions. Nodes are: 
* **SumNode** is a mixture model with components being either (ProductNode, List, or another SumNode)
* **ProductNode** is product of two (multivariate) random variables

The main advantage of SumProduct networks is that you can efficiently calculate the exact likelihood, marginals, and you can sample from them.

In this work, we add a **Dense** node, which supports a non-linear transformation of an input variable `x`, i.e. x = f(z) with the pdf transformed according to change of variables theorem. $$p(x) = \left|\frac{\partial f^{-1}(x)}{\partial x}\right| p(z)$$. In this implementation, `f` is a dense layer, i.e. $$f(x) = \sigma(W*x + b)$$. In order to be able to efficiently calculate the determinant of Jacobian and invert `f`, `W` is represent in its SVD decomposition as `W = UDV` where `U` and `V` are unitary and `D` is diagonal. Parametrized unitary matrices compatible with Flux / Zygote are provided in the package Unitary.jl.  Also note that the function `f` in **Dense** layer can be implemented by any flow-based models, such as *Residual Flows for Invertible Generative Modeling
Ricky T. Q. Chen, Jens Behrmann, David Duvenaud, Jörn-Henrik Jacobsen, 2019* and refenrences therein, but this is not implemented yet (PR welcomes).

The model is trained using the standard SGD and its variations

### God, give me dataset!!!   MNIST
```julia
using Flux, Flux.Data.MNIST, SumProductTransform, IterTools, StatsBase, Distributions, Unitary
using Images
using Base.Iterators: partition
using Flux: throttle, train!, Params
using EvalCurves, DrWatson, BSON

imgs = MNIST.images();
imgs = map(x -> imresize(x, ratio = 1/2), imgs);
X = Float32.(float(hcat(vec.(imgs)...)))
d = size(X,1)
noise = [0.5, 0.25, 0]
nc = [10,10,10]
model = buildmixture(d, nc,[identity, identity, identity], round.(Int,noise.*d));
tstidx = sample(1:size(X,2),100, replace = false)
xval = X[:,tstidx]

batchsize, iterations, maxpath = 100, 10000, 100
fit!(model, X, batchsize, iterations, maxpath, xval = xval);
```
If you want to sample from the trained model, just `rand(model)`.
