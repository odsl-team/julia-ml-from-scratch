# This file is licensed under the MIT License (MIT).

import Zygote

# Only compare fields present in both a and b, they must share at least one field:
function approx_cmp(a::T, b::U; kwargs...) where {T,U}
    if isstructtype(T) && fieldcount(T) > 0
        c = 0
        for n in fieldnames(T)
            if n in fieldnames(U)
                c = approx_cmp(getfield(a, n), getfield(b, n), kwargs...)  ? c + 1 : 0
            end
        end
        c > 0
    else
        isequal(a, b)
    end
end
approx_cmp(a::Number, b::Number; kwargs...) = isapprox(a, b; kwargs...)
approx_cmp(a::AbstractArray{<:Number}, b::AbstractArray{<:Number}; kwargs...) = isapprox(a, b; kwargs...)
approx_cmp(a::AbstractArray, b::AbstractArray; kwargs...) = all(map((x, y) -> approx_cmp(x, y; kwargs...), a, b))
approx_cmp(a::Tuple, b::Tuple; kwargs...) = all(map((x, y) -> approx_cmp(x, y; kwargs...), a, b))
approx_cmp(a::NamedTuple{names}, b::NamedTuple{names}; kwargs...) where names = approx_cmp(values(a), values(b); kwargs...)
approx_cmp(a::NamedTuple, b::NamedTuple; kwargs...) = false
approx_cmp(a, b::Nothing; kwargs...) = is_no_tangent(a)
approx_cmp(a::Nothing, b; kwargs...) = is_no_tangent(b)
is_no_tangent(::NoTangent) = true
is_no_tangent(x::NamedTuple) = is_no_tangent(values(x))
is_no_tangent(x::Union{Tuple,AbstractArray}) = all(is_no_tangent, x)


function test_pullback(f, xs...; kwargs...)
    dummy_dy = f(xs...)
    dx = pullback(dummy_dy, f, xs...)
    ref_dx = Zygote.pullback((f, xs...) -> f(xs...), f, xs...)[2](dummy_dy)
    approx_cmp(dx, ref_dx; kwargs...)
end
