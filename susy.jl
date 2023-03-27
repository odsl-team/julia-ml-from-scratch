# This file is licensed under the MIT License (MIT).

using LinearAlgebra, Statistics, Random
using StructArrays
using CompositionsBase
using Adapt
using Base: Fix1
using Base.Broadcast: BroadcastFunction
using Plots

using HDF5
input = h5open(joinpath(ENV["DATADIR"], "SUSY.hdf5"))

features = copy(transpose(read(input["features"])))
labels = Bool.(read(input["labels"]))

#=
using Random
# using LogExpFunctions, Distributions
using ChainRulesCore
using Base.Iterators: partition
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

# @functor LinearLayer

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
    BroadcastFunction(logistic),
    vec
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


f_xentroy_loss(labels::AbstractVector{Bool}) = mean ∘ Fix1(BroadcastFunction(xentropy), labels)



# =========================================================================================
# =========================================================================================

#stephist(Y, nbins = 100)


# Define loss:


current_loss = Base.Fix1(xentropy, L)

# cross-entropy is equivalent to negative mean likelihood:
loss(Y) ≈ mean(.- logpdf.(Bernoulli.(vec(Y)), L))


# Gradient calculation:

grad_model(model, loss, X) = Zygote.gradient((m,x) -> loss(m(x)), model, X)[1]

function loss_grad_model(model, loss, X)
    l, pullback = Zygote.pullback((m,x) -> loss(m(x)), model, X)
    d_model = pullback(one(l))[1]
    return l, d_model
end

grad_model(model, loss, X)
loss_grad_model(model, loss, X)


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
# optimizer = Adam(1e-4)

optimizer(model, grad_model(model, loss, X)) isa typeof(model)


# Split dataset:

L_train = L[begin:10000]
L_test = L[10001:end]
X_train = X[:,begin:10000]
X_test = X[:,10001:end]


# Train model, unbatched:

orig_model = deepcopy(model)

model = deepcopy(orig_model)
loss_train = Base.Fix1(xentropy, L_train)
loss_history = zeros(0)
optimizer = GradientDecent(0.025)
@showprogress for i in 1:1000
    l, d_model = loss_grad_model(model, loss_train, X_train)
    push!(loss_history, l)
    model = optimizer(model, d_model)
end
plot(loss_history)


# Train model, using batches and learning rate schedule:

model = deepcopy(orig_model)
loss_history = zeros(0)
for optimizer in GradientDecent.([0.1, 0.025, 0.01, 0.0025, 0.001, 0.00025])
        @showprogress for i in 1:250
        perm = shuffle(eachindex(L_train))
        shuffled_X = X_train[:, perm]
        shuffled_L = L_train[perm]
        batchsize = 200
        batch_loss_history = zeros(0)
        for idxs in partition(eachindex(shuffled_L), batchsize)
            X_batch = view(shuffled_X, :, idxs)
            L_batch = view(shuffled_L, idxs)
            loss_batch = Base.Fix1(xentropy, L_batch)
            l, d_model = loss_grad_model(model, loss_batch, X_batch)
            push!(batch_loss_history, l)
            model = optimizer(model, d_model)
        end
        push!(loss_history, mean(batch_loss_history))
    end
end
plot(loss_history)


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


#=
# Running on a GPU:

using CUDA
gpu_model = fmap(cu, model)
cu_loss = fmap(cu, loss)
gpu_X = cu(X)

grad_model(gpu_model, cu_loss, gpu_X)
loss_grad_model(gpu_model, cu_loss, gpu_X)
optimizer(gpu_model, grad_model(gpu_model, cu_loss, gpu_X))

@benchmark loss_grad_model($model, $loss, $X)
@benchmark loss_grad_model($gpu_model, $cu_loss, $gpu_X)
=#
