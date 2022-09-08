using Flux
using Mill
using StatsBase
using PoissonRandom
using SpecialFunctions


abstract type AbsDist{T<:Real} end
const _BagNode{T} = BagNode{ArrayNode{Matrix{T}, N}, S, N} where {N<:Nothing, S<:AbstractBags{Int64}}


mutable struct _Poisson{Tr} <: AbsDist{Tr}
    v::AbstractArray{Tr,1}
end
Flux.@functor _Poisson
function logpdf(m::_Poisson{T}, n::Int) where {T<:Real}
    v = exp.(m.v)
    n*log.(v) - v .- T(logfactorial(n))
end
drawobs(m::_Poisson, n::T) where {T<:Int} = map(_->pois_rand(ceil(T, first(m.v))), 1:n)


mutable struct _MvNNormal{Tr,Ti} <: AbsDist{Tr}
    d::Ti
    a::AbstractArray{Tr,1}
    b::AbstractArray{Tr,1}
end
Flux.@functor _MvNNormal
function logpdf(m::_MvNNormal{T}, x::AbstractArray{T,2}) where {T<:Real}
    z = m.a.*x .+ m.b
    -T(0.5)*(m.d*log(T(2.0)*T(pi)) .+ sum(z.^2, dims=1)) .+ sum(log.(m.a)) .+ eps(T)
end
drawobs(m::_MvNNormal{T}, n::Int) where {T<:Real} = m.b .+ m.a.*randn(T, m.d, n)


mutable struct ProcessNode{T}
    c::AbsDist{T}
    f::AbsDist{T}
end
Flux.@functor ProcessNode
function ProcessNode{Tr}(d::Ti) where {Tr<:Real,Ti<:Int}
    ProcessNode(_Poisson(ones(Tr, 1)), _MvNNormal(d, ones(Tr, d), randn(Tr, d)))
end
function logpdf(m::ProcessNode{T}, x::_BagNode{T}) where {T<:Real}
    p_inst = logpdf(m.f, x.data.data)
    p_bags = mapreduce(b->logpdf(m.c, length(b)) + sum(p_inst[:, b], dims=2), hcat, x.bags)
    return p_bags
end
function drawobs(m::ProcessNode{Tr}, nbags::Ti) where {Tr<:Real,Ti<:Int}
    v = drawobs(m.c, nbags)
    x = map(v->drawobs(m.f, v), v)
    hcat(x...), mapreduce((x, i)->repeat([i], size(x, 2)), vcat, x, 1:nbags)
end


function main()
    d = 2
    s = 0f0
    nbags = 200

    m = ProcessNode{Float32}(d)
    x = BagNode(drawobs(m, nbags)...)

    logpdf(m, x)

    ps = Flux.params(m)
    gs = gradient(()->sum(logpdf(m, x)), ps)

    display(ps)
    foreach(p->display(gs[p]), ps)
end

main()
