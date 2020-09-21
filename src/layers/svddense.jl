using Unitary 
using LinearAlgebra
using NNlib
using Flux

################################################################################
#                   (non-linear) Affine transformations                        #
#           Ref: Sum-Product-Transform Networks: Exploiting Symmetries         #
#                        using Invertible Transformations                      #
#   Tomas Pevny, Vasek Smidl, Martin Trapp, Ondrej Polacek, Tomas Oberhuber    #               #
#                          arxiv.org/abs/2005.01297                            #
################################################################################

"""
    SVDDense

    Dense layer with square weight matrix of dimension `n` parametrized in 
    SVD decomposition 
    
    `σ` --- an invertible and transfer function, cuurently implemented 'leakyrelu`, `tanh`, `selu` and `identity`
"""
struct SVDDense{U, D<:DiagonalRectangular, V, B, S} <: Bijector{1}
    u::U
    d::D
    v::V
    b::B
    σ::S
end

Base.show(io::IO, m::SVDDense) = print(io, "SVDDense{$(size(m.d)), $(m.σ)}")

Flux.@functor SVDDense

"""
    SVDDense(n, σ; indexes = :random)

    Dense layer with square weight matrix of dimension `n` parametrized in 
    SVD decomposition using `UnitaryGivens`  parametrization of unitary matrix.
    
    `σ` --- an invertible and transfer function, cuurently implemented `selu` and `identity`
    indexes --- method of generating indexes of givens rotations (`:butterfly` for the correct generation; `:random` for randomly generated patterns)
"""
function SVDDense(n::Int, σ = identity, unitary = :butterfly)
    n == 1 && return(ScaleShift(1, σ))
    if unitary == :householder
        return(_svddense_householder(n, σ))
    elseif unitary == :butterfly || unitary == :givens
        return(_svddense_butterfly(n, σ))
    else 
        @error "unknown type of unitary matrix $unitary"
    end
end

_svddense_butterfly(n::Int, σ) = 
    SVDDense(Givens(n), 
            DiagonalRectangular(rand(Float32,n), n, n),
            Givens(n),
            0.01f0.*randn(Float32,n),
            σ)

_svddense_householder(n::Int, σ) = 
    SVDDense(UnitaryHouseholder(Float32, n), 
            DiagonalRectangular(rand(Float32,n), n, n),
            UnitaryHouseholder(Float32, n) ,
            0.01f0.*randn(Float32,n),
            σ)

function _transform(m::SVDDense, x)
    z = m.u * (m.d * (m.v * x)) .+ m.b
    (transformed = m.σ.(z), z = z)
end

(m::SVDDense)(x::Unitary.AbstractMatVec) = _transform(m, x).transformed


function forward(m::SVDDense, x::AbstractVecOrMat)
    transformed, z = _transform(m, x)

    g = _explicitgrad.(m.σ, z)
    logabsdetjac = sum(log.(g), dims = 1) .+ _logabsdet(m.d)

    return (rv = transformed, logabsdetjac = logabsdetjac)
end

logabsdetjac(m::SVDDense, x) = forward(m, x).logabsdetjac


struct InvertedSVDDense{U, D, V, B, S} <: Bijector{1}
    u::U
    d::D
    v::V
    b::B
    σ::S
end
Flux.@functor InvertedSVDDense

Base.show(io::IO, m::InvertedSVDDense) = print(io, "InvertedSVDDense{$(size(m.d)), $(m.σ)}")


function _transform(m::InvertedSVDDense, x)
    m.v * (m.d * (m.u * (m.σ.(x) .- m.b)))
end

(m::InvertedSVDDense)(x::Unitary.AbstractMatVec) = _transform(m, x)

function forward(m::InvertedSVDDense, x::AbstractVecOrMat)
    transformed= _transform(m, x)

    g = _explicitgrad.(m.σ, x)
    logabsdetjac = sum(log.(g), dims = 1) .+ Unitary._logabsdet(m.d)

    return (rv = transformed, logabsdetjac = logabsdetjac)
end

logabsdetjac(m::InvertedSVDDense, x) = forward(m, x).logabsdetjac

Base.inv(m::SVDDense) = InvertedSVDDense(inv(m.u), inv(m.d), inv(m.v), m.b, inv(m.σ))
Base.inv(m::InvertedSVDDense) = SVDDense(inv(m.u), inv(m.d), inv(m.v), m.b, inv(m.σ))

isclosedform(b::Bijectors.Inverse{<:SVDDense}) = false