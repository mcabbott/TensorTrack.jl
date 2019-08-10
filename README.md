# TensorTrack.jl

[![Build Status](https://travis-ci.org/mcabbott/TensorTrack.jl.svg?branch=master)](https://travis-ci.org/mcabbott/TensorTrack.jl)

This package is one approach to adding gradient definitions to [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)
for reverse-mode auto-differentiation. Originally for [Tracker.jl](https://github.com/FluxML/Tracker.jl), 
now also for [Zygote.jl](https://github.com/FluxML/Zygote.jl).

Everything `TensorOperations` calculates comes down to three basic functions: 
`contract!` (generalised matrix multiplication) `trace!` (traces!) and `add!` (index permutations, and addition).
This package simply overloads each of them to compute the sensitivities of their inputs 
in terms of other such functions. Here's the simplest one:
```julia
Tracker.@grad function add!(α, A, conjA, β, C, indCinA)
    add!(data(α), data(A), conjA, data(β), data(C), indCinA),
    Δ -> ∇add(Δ, α, A, conjA, β, C, indCinA)
end

function ∇add(Δ, α::Tα, A::TA, conjA, β::Tβ, C::TC, indCinA) where {Tα,TA,Tβ,TC}
    ∇A = TA<:TrackedArray ? add∇A(data(Δ), data(α), A, conjA, β, C, indCinA) : nothing
    ∇C = TC<:TrackedArray ? data(β) .* data(Δ) : nothing
    ∇α = 0
    ∇β = 0
    return (∇α, ∇A, nothing, ∇β, ∇C, nothing)
end

function add∇A(Δ, α, A::TA, conjA, β, C::TC, indCinA) where {TA,TC}
    add!(α, Δ, conjA, 0, similar(data(A)), invperm(indCinA))
end
```
The only hard part of this is getting all the things like `invperm(indCinA)` straight.

### Limitations:
1. It will tend to compute too many gradients
  (especially with Zygote where it cannot ask `TA<:TrackedArray ?` before calculating).
2. It does not handle constants like `α * B[i,j]` at all (hence `∇α = 0` above).
3. Since this is entirely within TensorOperations.jl, it cannot handle non-Einstein contractions
  such as `A[i,k] * B[j,k] * C[l,k]`. 
4. It depends on many internal details of TensorOperations.jl, so will break if that package changes.

See also [TensorGrad.jl](https://github.com/mcabbott/TensorGrad.jl) for another approach, 
at the level of the macro not the functions. 

--- Michael Abbott, January 2019
