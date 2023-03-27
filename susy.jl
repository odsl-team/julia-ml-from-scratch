# This file is licensed under the MIT License (MIT).

using LinearAlgebra, Statistics, Random

using Base: Fix1
using Base.Broadcast: BroadcastFunction
using Base.Iterators: partition

using StructArrays
using CompositionsBase
using Adapt

using Functors: @functor, functor, fmap

using Plots
import ProgressMeter

using HDF5
input = h5open(joinpath(ENV["DATADIR"], "SUSY.hdf5"))

features = copy(transpose(read(input["features"])))
labels = Bool.(transpose(read(input["labels"])))

#=
using Random
# using LogExpFunctions, Distributions
using ChainRulesCore
using Functors: @functor, functor, fmap
using Plots, BenchmarkTools, ProgressMeter
=#


struct NoTangent end
Base.sum(::AbstractArray{<:NoTangent}) = NoTangent()


pullback(dy, ::typeof(*), a, b) = (NoTangent(), dy * b', a' * dy)


function pullback(dy, f::ComposedFunction, x)
    tmp = f.inner(x)
    d_outer, d_tmp = pullback(dy, f.outer, tmp)
    d_inner, dx = pullback(d_tmp, f.inner, x)
    return ((outer = d_outer, inner = d_inner), dx)
end


function pullback(dy, f::Fix1, x2)
    dff, dfx, dx2 = pullback(dy, f.f, f.x, x2)
    return ((f = dff, x = dfx), dx2)
end


_pullback_for_bc(dy_tpl, args...) = pullback(dy_tpl[1], args...)

function pullback(dY, bf::BroadcastFunction, Xs...)
    # Require all inputs to have the same shape, to simplify things:
    all(isequal(size(first(Xs))), map(size, Xs))

    # Wrap dY in StructArray to generate StructArray result in broadcast:
    dY_sa = StructArray((dY,))
    tangents_sa = broadcast(_pullback_for_bc, dY_sa, bf.f, Xs...)
    tangents = StructArrays.components(tangents_sa)

    return (sum(first(tangents)), Base.tail(tangents)...)
end


pullback(dy, ::typeof(vec), x) = NoTangent(), reshape(dy, size(x)...)


pullback(dy, ::typeof(sum), x) = NoTangent(), fill!(similar(x), dy)
pullback(dy, ::typeof(mean), x) = NoTangent(), fill!(similar(x), dy / length(x))




struct LogCalls{F} <: Function
    f::F
end

@functor LogCalls

function (lf::LogCalls)(xs...)
    @info "primal $(lf.f)"
    lf.f(xs...)
end

function pullback(dy, lf::LogCalls, xs...)
    @info "pullback $(lf.f)"
    pullback(dy, lf.f, xs...)
end




struct LinearLayer{
    MA<:AbstractMatrix{<:Real},
    VB<:AbstractVector{<:Real}
} <: Function
    A::MA
    b::VB
end

@functor LinearLayer

(f::LinearLayer)(x::AbstractVecOrMat{<:Real}) = f.A * x .+ f.b

function pullback(dy, f::LinearLayer, x)
    _, dA, dx = pullback(dy, *, f.A, x)
    db = vec(sum(dy, dims = 2))
    ((A = dA, b = db), dx)
end


function glorot_uniform!(rng::AbstractRNG, A::AbstractMatrix{T}, gain::Real = one(T)) where {T<:Real}
    fan_in_plus_fan_out = sum(size(A))
    scale = sqrt(T(24) / fan_in_plus_fan_out)
    rand!(rng, A)
    A .= T(gain) .* scale .* (A .- T(0.5))
    return A
end


function glorot_normal!(rng::AbstractRNG, A::AbstractMatrix{T}, gain::Real = one(T)) where {T<:Real}
    fan_in_plus_fan_out = sum(size(A))
    scale = sqrt(T(2) / fan_in_plus_fan_out)
    rand!(rng, A)
    A .= gain .* scale .* A
    return A
end


function LinearLayer(rng::AbstractRNG, n_in::Integer, n_out::Integer, f_init! = glorot_uniform!, ::Type{T} = Float32) where {T<:Real}
    A = Matrix{T}(undef, n_out, n_in)
    f_init!(rng, A)
    b = similar(A, n_out)
    fill!(b, zero(T))
    return LinearLayer(A, b)
end



relu(x::Real) = max(zero(x), x)

pullback(dy, ::typeof(relu), x) = NoTangent(), ifelse(x > 0, dy, zero(dy))


logistic(x::Real) = inv(exp(-x) + one(x))

function pullback(dy, ::typeof(logistic), x)
    z = logistic(x)
    return NoTangent(), dy * z * (1 - z)
end





rng = Random.default_rng()

model = opcompose(
    LinearLayer(rng, 18, 128),
    BroadcastFunction(relu),
    LinearLayer(rng, 128, 128),
    BroadcastFunction(relu),
    LinearLayer(rng, 128, 1),
    BroadcastFunction(logistic)
)


X = rand(Float32, 18, 1000)

Y = model(X)
dY = rand(Float32, size(Y)...)

pullback(dY, model, X)


xentropy(label::Bool, output::Real) = - log(ifelse(label, output, 1 - output))

#=
using Distributions
xentropy(true, 0.3) ≈ - loglikelihood(Bernoulli(0.3), true)
xentropy(false, 0.3) ≈ - loglikelihood(Bernoulli(0.3), false)
=#


pullback(dy, ::typeof(xentropy), label, output) = NoTangent(), NoTangent(), - dy / ifelse(label, output, output - 1)


f_xentroy_loss(labels) = mean ∘ Fix1(BroadcastFunction(xentropy), labels)



# Split dataset:

idxs_total = eachindex(labels)
n_total = length(idxs_total)
n_train = round(Int, 0.7 * n_total)
idxs_train = 1:n_train
idxs_test = n_train+1:n_total

#ArrayType = Array

using CUDA
ArrayType = CuArray

#=
using Metal
ArrayType = MtlArray
=#

m = adapt(ArrayType, model)
X_all = adapt(ArrayType, features)
L_all = adapt(ArrayType, labels)

L_train = view(L_all, :, idxs_train)
L_test = view(L_all, :, idxs_test)
X_train = view(X_all, : ,idxs_train)
X_test = view(X_all, :, idxs_test)


# Manual batch evaluation

batchsize = 50000
shuffled_idxs = shuffle(rng, eachindex(L_train))
partitions = partition(adapt(ArrayType, shuffled_idxs), batchsize)
idxs = first(partitions)
L = L_train[:, idxs]
X = X_train[:, idxs]

f_loss = f_xentroy_loss(L)
f_model_loss = f_loss ∘ m

l = f_model_loss(X)
grad_model_loss = pullback(one(Float32), f_model_loss, X)

#=
using BenchmarkTools
@benchmark $f_model_loss($X)
@benchmark pullback(one(Float32), $f_model_loss, $X)
=#

f_model_loss.inner == m
grad_model = grad_model_loss[1].inner


# Define gradient descent optimizer:

struct GradientDecent{T}
    rate::T
end

(opt::GradientDecent)(x, ::Nothing) = x
(opt::GradientDecent)(x::Real, dx::Real) = x - opt.rate * dx
(opt::GradientDecent)(x::AbstractArray, dx::AbstractArray) = x .- opt.rate .* dx
function (opt::GradientDecent)(x, dx)
    content_x, re = functor(x)
    content_dx, _ = functor(dx)
    re(map(opt, content_x, content_dx))
end


optimizer = GradientDecent(1)
# optimizer = ADAM(1e-4)

optimizer(m, grad_model) isa typeof(m)


# Train model, using batches and learning rate schedule:
m_trained = deepcopy(m)

n_epochs = 2
batchsize = 50000
optimizer = GradientDecent(0.1)

loss_history = zeros(0)
n_batches = length(partition(axes(L_train, 2), batchsize))
p = ProgressMeter.Progress(n_epochs * n_batches, 0.1, "Training...")
for epoch in 1:n_epochs
    shuffled_idxs = shuffle(rng, axes(L_train, 2))
    partitions = partition(adapt(ArrayType, shuffled_idxs), batchsize)

    idxs = first(partitions)
    
    batch_loss_history = zeros(0)
    for idxs in partitions
        L = L_train[:, idxs]
        X = X_train[:, idxs]

        f_loss = f_xentroy_loss(L)
        f_model_loss = f_loss ∘ m_trained
        
        loss_current_batch = f_model_loss(X)
        push!(loss_history, loss_current_batch)
        grad_model_loss = pullback(one(Float32), f_model_loss, X)
        grad_model = grad_model_loss[1].inner

        m_trained = optimizer(m_trained, grad_model)

        ProgressMeter.next!(p; showvalues = [(:loss_train, loss_current_batch),#= (:loss_test, loss_test)=#])
    end
    #push!(loss_history, mean(batch_loss_history))
end
ProgressMeter.finish!(p)
plot(loss_history)




# =======================================================================

# Evaluate trained model:

Y = model(X)
threshold = 0:0.01:1
TPR = [count((Y .>= t) .&& L) / count(L) for t in threshold]
FPR = [count((Y .>= t) .&& .! L) / count(L) for t in threshold]
Y_thresh = Y .>= 0.5

plot(
    begin
        stephist(L, nbins = 100, normalize = true, label = "Truth")
        stephist!(model(X_train), nbins = 100, normalize = true, label = "Training pred.")
        stephist!(model(X_test), nbins = 100, normalize = true, label = "Test pred.")
    end,
    begin
        plot(threshold, TPR, label = "TPR", color = :green, xlabel = "treshold")
        plot!(threshold, FPR, label = "FPR", color = :red)
    end,
    plot(FPR, TPR, label = "ROC", xlabel = "FPR", ylabel = "TPR"),
    begin
        stephist(edep, nbins = 1500:5:1700, label = "all", xlabel = "E [keV]")
        stephist!(edep[findall(L)], nbins = 1500:5:1700, label = "label SSE")
        stephist!(edep[findall(Y_thresh)], nbins = 1500:5:1700, label = "model SSE")
    end
)
