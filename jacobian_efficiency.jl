# This software is licensed under the MIT "Expat" License.

using Base.Threads: Atomic, atomic_add!

onehot(i::Integer, n::Integer) = [i == j for j in 1:n]


# A special number type that counts operations:

struct OpsReal{T<:Real} <: Real
    x::T
end

const opscount = Base.Threads.Atomic{Int}(0)
clearops() = (opscount[] = 0; nothing)
getops() = Base.Threads.atomic_xchg!(opscount, 0)

convert(::Type{OpsReal{T}}, x::Real) where T = OpsReal{T}(x)
convert(::Type{OpsReal}, x::Real) = OpsReal(x)

Base.:+(a::OpsReal, b::OpsReal) = (atomic_add!(opscount, 1); OpsReal(a.x + b.x))
Base.:*(a::OpsReal, b::OpsReal) = (atomic_add!(opscount, 1); OpsReal(a.x * b.x))

Base.promote_rule(::Type{OpsReal{T}}, ::Type{U}) where {T<:Real,U<:Real} = OpsReal{promote_type(T,U)}

OpsReal(1.0) * 2 + 3
getops()


# Let's assume we have a f_loss = f∘g∘h, so a computation f(g(h(θ))). The
# Jacobian of f_loss is therefore J_f * J_g * J_h.

J_f, J_g, J_h = OpsReal.(rand(1,3)), OpsReal.(rand(3,4)), OpsReal.(rand(4,5))

# Since f_loss maps many parameters to a single loss value, the whole
# Jacobian has a single row:

J_f * J_g * J_h

# Let's start counting operations.

clearops()

# Since we run f(g(h(x))), we first get J_h, then J_g, and finally J_f. So
# it's memory-efficient to compute the whole Jacobian in this order.

J_f * (J_g * J_h)
getops()

# But that's expensive - compare with

(J_f * J_g) * J_h
getops()

# That comes with a price though - now we need to keep all intermediate
# Jacobians in memory.

# If we don't want to have full Jacobians in memory at all, but can directly
# use the state of the computation to calculate products `J * x`, then we can
# retrieve each columns (here of length one) of the total Jacobian separately:

[J_f * (J_g * (J_h * onehot(i, 5))) for i in 1:5]

# That is what forward-mode automatic differentiation does. It gives us the
# gradient and it's very memory-efficient, but also very expensive for
# many-to-one problems:

getops()

# But if we store enough of the intermediate states of computation, so that we
# can left-multiply a one with the last Jacobian, then multiply with the
# second-to-last, and so on - then we only have to perform products of
# row-vectors with Jacobians:

((1 * J_f) * J_g) * J_h

# This is what reverse-mode automatic differentiation does. It's more
# memory-intensive, but computationally very efficient for many-to-one
# problems:

getops()

# Alternatively we can write `(J_h' * J_g' * J_f' * 1)'`, so in reverse-mode
# AD we implement products of adjoint Jacobian operators with vectors.
