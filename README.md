# SumProductTransform.jl

Is an experimental package for experimenting with SumProductTransformation networks (their main advantage is the exact calculation of likelihood). The package puts emphasis on flexibility, which means that it is not super fast, but can be safely used for flexible experimentation. It has been created as a testbed for the paper *Sum-Product-Transform Networks: Exploiting Symmetries using Invertible Transformations, Tomas Pevny, Vasek Smidl, Martin Trapp, Ondrej Polacek, Tomas Oberhuber, 2020* [https://arxiv.org/abs/2005.01297](https://arxiv.org/abs/2005.01297)

For an example of GMM, SPN, and SPTN you can see a Pluto notebook in `examples/pluto.jl`.

**The package depends on** [https://github.com/pevnak/Unitary.jl](https://github.com/pevnak/Unitary.jl) which is not registered, as is not this package `SumProductTransform`.

An experimental implementation of a generalization of a Sum-Product networks by a Dense node.

*Background:* The Sum-Product-Transform networks is a hierarchical model with a tree structure composed by following nodes: 
* **LeafNode** is a known tractable probabilisty distribution, usually a multivariate normal distribution.
* **SumNode** is a mixture model with components being either (ProductNode, List, or another SumNode);
* **ProductNode** is product of random variables assuming their independency;
* **TransformationNode** implements a change of variables formula `x = f(z)` with the pdf transformed according to change of variables theorem. ![p(x) = \left|\frac{\partial f^{-1}(x)}{\partial x}\right| p(z)](/docs/change.svg).

The change of variables in TransformationNode can encapsulate anything which allows calculation of `logabsdet` (e.g. flow models), but we prefer to implement it as a dense layer, i.e. ![f(x) = \sigma(W*x + b)](/docs/dense.svg), where `W` is a square matrics. In order to be able to efficiently calculate the determinant of Jacobian and invert `f`, `W` is represent in its SVD decomposition as `W = UDV` where `U` and `V` are unitary and `D` is diagonal. Group of Unitary matrices parametrized in a gradient descend friendly way are provided in the package https://github.com/pevnak/Unitary.jl


### A commented example of GMM, SPN, and SPTN

```julia
using ToyProblems, SumProductTransform, Unitary, Flux, Setfield
using ToyProblems: flower2
using DistributionsAD: TuringMvNormal
using SumProductTransform: ScaleShift, SVDDense

x = flower2(Float32, 1000, npetals = 9)
```



To create a Gaussian Mixture Model with 9 components and Normal distribution on leaves with full covariance, we use a single sumnodes with `TuringMvNormal` transformed by Affine distribution `SVDDense(d)`  (MvNormal with full covariance `TuringDenseMvNormal` does not support Zygote, therefor this construction is prefered). This is a way for us to implement general normal distribution. If you fancy a normal distribution with non-zeros only on diagonal, use `ScaleShift(d)` instead of `SVDDense(d).` To fit the model on data `x` use `fit!` function. 

```
d = size(x,1)
ncomponents = 9
model = SumNode([TransformationNode(SVDDense(d), TuringMvNormal(d, 1f0)) for i in 1:ncomponents])
batchsize = 100
nsteps = 20000
history = fit!(model, x, batchsize, nsteps)
```

To calculate the loglikelihood on samples `x` use `logpdf(model, x)` and to sample from the model, use `rand(model)`.

To create a simple SumProductNetwork, we can do

```
components = map(1:ncomponents) do _
  p₁ = SumNode([TransformationNode(ScaleShift(1), TuringMvNormal(1, 1f0)) for _ in 1:ncomponents])
  p₂ = SumNode([TransformationNode(ScaleShift(1), TuringMvNormal(1, 1f0)) for _ in 1:ncomponents])
  p₁₂ = ProductNode((p₁, p₂))
end
model = SumNode(components)
```
and you can fit it the same way as above.

Finally, to create a SumProductTransform network, you can do

```
ncomponents = 3
nlayers = 3
model = TransformationNode(ScaleShift(d),  TuringMvNormal(d,1f0))
for i in 1:nlayers
  model = SumNode([TransformationNode(SVDDense(2), model) for i in 1:ncomponents])
end
```


### Compatibility with Flux / Zygote
The model is compatible with Flux / Zygote. So you can take parameters (weights in SumNodes, parameters of TransformationNodes), you just hit `ps = Flux.params(model)` and the gradient of `logpdf` is differentiable as `gradient(() -> logpdf(model, x), ps)`. The `fit!` is an optimized version of `train!` function which utilizes threading. 

### Compatibility with Bijectors.jl
`SVDDense`, `ScaleShift`, and `LUDense` implement the interface of `Bijectors.jl`, which means that you can use them the same way. **Caution!!!*** If you want to use some layers from bijectors, verify that `Flux.params` returns parameters.

For example:
```
julia> using Flux
julia> using Bijectors
julia> Flux.params(PlanarLayer(2))
Params([])
```
but if you register the layer with `Flux`
```
 Flux.@functor PlanarLayer
 Flux.params(PlanarLayer(2))
 ```
 magic happens. 
 This allows you to use `PlanarLayer` within SumProductTransform as 
 ```
ncomponents = 3
nlayers = 3
model = TransformationNode(PlanarLayer(d),  TuringMvNormal(d,1f0))
for i in 1:nlayers
  model = SumNode([TransformationNode(PlanarLayer(2), model) for i in 1:ncomponents])
end

history = fit!(model, x, 100, 20000)
```


