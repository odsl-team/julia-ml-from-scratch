# *This file is licensed under the MIT License (MIT).*

# ## HEP ML from scratch in Julia
#
# MPP IMPRS / TUM / ORIGINS Data Science Block Course
#
# March 2023, Oliver Schulz <oschulz@mpp.mpg.de>

# ### Julia project environment

# Check that the right Julia project environment is active:

using Pkg
basename(dirname(Pkg.project().path))


# ### Dependencies

# Julia standard libary functionality:

using LinearAlgebra, Statistics, Random

using Base: Fix1
using Base.Broadcast: BroadcastFunction
using Base.Iterators: partition

# Some data structure packages we need:

using Adapt, StructArrays, ConstructionBase

# Plotting and I/O packages:

using Plots, ProgressMeter
using HDF5

# ### Configuration options
#
# Use the environment variable `$DATADIR` to specify a different data location than the directory of this scipt/notebook.
# Set `$COMPUTE_BACKEND` to "CUDA" or "METAL" to use the NVIDIA CUDA or Apple Metal compute backends, respectively.
# Note: The Julia Metal implementation still has some issues, there may be problems with latency and memory overflow.

datadir = get(ENV, "DATADIR", @__DIR__)
compute_backend = get(ENV, "COMPUTE_BACKEND", "CPU")


maybe_copy(A::AbstractArray) = A

if compute_backend == "CPU"
    ArrayType = Array
elseif compute_backend == "CUDA"
    using CUDA
    ArrayType = CuArray
elseif compute_backend == "METAL"
    using Metal
    ArrayType = MtlArray

    #See https://github.com/JuliaGPU/Metal.jl/issues/149:
    maybe_copy(A::SubArray{<:Real, N, <:MtlArray}) where N = copy(A)
else
    error("Unsupported compute backend: $compute_backend")
end


# ### Dataset

# We'll use the SUSY Data Set from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SUSY).
#
# Note: Use the script "hepmldata_to_hdf5.jl" to convert the original "SUSY.csv.gz" to "SUSY.hdf5".

input = h5open(joinpath(datadir, "SUSY.hdf5"))
features = copy(transpose(read(input["features"])))
labels = Bool.(transpose(read(input["labels"])))


# ### A simple automatic differentiation system

struct NoTangent end
const δ∅ = NoTangent()
Base.sum(::AbstractArray{<:NoTangent}) = δ∅


pullback(δy, ::typeof(*), a, b) = (δ∅, δy * b', a' * δy)

pullback(δy, ::typeof(sum), x) = δ∅, fill!(similar(x), δy)
pullback(δy, ::typeof(mean), x) = δ∅, fill!(similar(x), δy / length(x))


function pullback(δz, fg::ComposedFunction, x)
    f, g = fg.outer, fg.inner
    #Chain-rule: δx == ( δz * J_f(y) ) * J_g(x)
    y = g(x)
    δf, δy = pullback(δz, f, y)
    δg, δx = pullback(δy, g, x)
    return ((outer = δf, inner = δg), δx)
end


function pullback(δy, f::Fix1, x2)
    δf_f, δf_x, δx2 = pullback(δy, f.f, f.x, x2)
    return ((f = δf_f, x = δf_x), δx2)
end


_bcf_pullback_kernel(δy_ntpl1, args...) = pullback(only(δy_ntpl1), args...)

function pullback(δY, bf::BroadcastFunction, Xs...)
    #Require all inputs to have the same shape, to simplify things:
    @assert all(isequal(size(first(Xs))), map(size, Xs))

    #Wrap δY in StructArray with eltype NTuple{1} to make broadcast return a
    #StructArray. Easy GPU-compatible way to separate tangent components:
    δY_ntpl1_sa = StructArray((δY,))
    δ_f_Xs_sa = broadcast(_bcf_pullback_kernel, δY_ntpl1_sa, bf.f, Xs...)
    δ_f_Xs = StructArrays.components(δ_f_Xs_sa)

    return (sum(first(δ_f_Xs)), Base.tail(δ_f_Xs)...)
end


# ### Utility functions

# Utility to log function calls:

struct LogCalls{F} <: Function
    f::F
end

function (lf::LogCalls)(xs...)
    @info "primal $(lf.f)"
    lf.f(xs...)
end

function pullback(δy, lf::LogCalls, xs...)
    @info "pullback $(lf.f)"
    pullback(δy, lf.f, xs...)
end



# ### Definition of a linear NN layer
# 
# A linear neural network layer that implements the function
# $$A * x + b$$

# A struct for the layer, make it a subtype of `Function`:

struct LinearLayer{
    MA<:AbstractMatrix{<:Real},
    VB<:AbstractVector{<:Real}
} <: Function
    A::MA
    b::VB
end

# Define what it does:

(f::LinearLayer)(x::AbstractVecOrMat{<:Real}) = f.A * x .+ f.b

# Define the pullback:

function pullback(δy, f::LinearLayer, x)
    _, δA, δx = pullback(δy, *, f.A, x)
    δb = vec(sum(δy, dims = 2))
    ((A = δA, b = δb), δx)
end

# Glorot weight initialization schemes (with uniform and normal distribution):

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

# Convenience constructor:

function LinearLayer(rng::AbstractRNG, n_in::Integer, n_out::Integer, f_init! = glorot_uniform!, ::Type{T} = Float32) where {T<:Real}
    A = Matrix{T}(undef, n_out, n_in)
    f_init!(rng, A)
    b = similar(A, n_out)
    fill!(b, zero(T))
    return LinearLayer(A, b)
end



# ### Activation functions

# RELU activation:

relu(x::Real) = max(zero(x), x)

pullback(δy, ::typeof(relu), x) = δ∅, ifelse(x > 0, δy, zero(δy))


# Logistic (sigmoid) activation:

logistic(x::Real) = inv(exp(-x) + one(x))

function pullback(δy, ::typeof(logistic), x)
    z = logistic(x)
    return δ∅, δy * z * (1 - z)
end


# ### Model definition

rng = Random.default_rng()

# Define a simple model with 3 layers. The layers are stored in a tuple here
# in order of application:

model_layers = (
    LinearLayer(rng, 18, 128),
    BroadcastFunction(relu),
    LinearLayer(rng, 128, 128),
    BroadcastFunction(relu),
    LinearLayer(rng, 128, 1),
    BroadcastFunction(logistic)
)

# Use Julia's function composition operator ∘ to chain the layers.
#
# Note: `(f∘g∘h)(x) === f(g(h(x)))`, so reverse layer order before composing:

model = ∘(reverse(model_layers)...)


# Let's try the model on some random data:

X = rand(Float32, 18, 1000)
Y = model(X)

#-

δY = rand(Float32, size(Y)...)

#-

pullback(δY, model, X)


# ### Log-Likelihood functions

# Without this, things will become numerically unstable with 32-bit float precision:

force_nz(x::T) where {T<:Real} = ifelse(iszero(x), eps(T), x)

# Binary cross-entropy:

function binary_xentropy(label::Bool, output::Real)
    - log(force_nz(ifelse(label, output, 1 - output)))
end

function pullback(δy, ::typeof(binary_xentropy), label, output::Real)
    δ∅, δ∅, - δy / force_nz(ifelse(label, output, output - 1))
end

# The binary cross-entroy is the log-likelihood of a Bernoulli distribution
# (uncomment to check):

#using Distributions
#binary_xentropy(true, 0.3) ≈ - logpdf(Bernoulli(0.3), true)

#-

#binary_xentropy(false, 0.3) ≈ - logpdf(Bernoulli(0.3), false)


# `f_loglike_loss` will generate a loss function, given a log-likelihood
# function and truth/labels:

f_loglike_loss(f_loglike, labels::AbstractArray) = mean ∘ Fix1(BroadcastFunction(f_loglike), labels)


# Training and test dataset

idxs_total = eachindex(labels)
n_total = length(idxs_total)
n_train = round(Int, 0.7 * n_total)
idxs_train = 1:n_train
idxs_test = n_train+1:n_total

m = adapt(ArrayType, model)
X_all = adapt(ArrayType, features)
L_all = adapt(ArrayType, labels)

L_train = maybe_copy(view(L_all, :, idxs_train))
L_test = maybe_copy(view(L_all, :, idxs_test))
X_train = maybe_copy(view(X_all, : ,idxs_train))
X_test = maybe_copy(view(X_all, :, idxs_test))

typeof(X_all)


# Manual evaluation of a single batch

batchsize = 50000
shuffled_idxs = shuffle(rng, eachindex(L_train))
partitions = partition(adapt(ArrayType, shuffled_idxs), batchsize)
idxs = first(partitions)
L = L_train[:, idxs]
X = X_train[:, idxs]

f_loss = f_loglike_loss(binary_xentropy, L)
f_model_loss = f_loss ∘ m

model_loss = f_model_loss(X)

#-

grad_model_loss = pullback(one(Float32), f_model_loss, X)

# The inner function of `f_loss ∘ m` is `m`:

f_model_loss.inner == m

# So we can retrieve the gradient of `m` via

grad_model = grad_model_loss[1].inner


# How fast are we (uncomment for performance benchmark):

#using BenchmarkTools
#@benchmark $f_model_loss($X)

#-

#@benchmark pullback(one(Float32), $f_model_loss, $X)


# ### Optimizer implementation

# Let's define an `AbstractOptimizer` type to keeps things extensible:

abstract type AbstractOptimizer end

apply_opt(::AbstractOptimizer, x, ::Union{NoTangent, Nothing}) = x
apply_opt(opt::AbstractOptimizer, x::NTuple{N}, Δx::NTuple{N}) where N = map((xi, dxi) -> apply_opt(opt, xi, dxi), x, Δx)
apply_opt(opt::AbstractOptimizer, x::NamedTuple{names}, Δx::NamedTuple{names}) where names = map((xi, dxi) -> apply_opt(opt, xi, dxi), x, Δx)
apply_opt(opt::AbstractOptimizer, x::T, Δx) where T = constructorof(T)(values(apply_opt(opt, getfields(x), Δx))...)


# A simple gradient decent optimizer with fixed learning rate:

struct GradientDecent{T} <: AbstractOptimizer
    rate::T
end

apply_opt(opt::GradientDecent, x::T, Δx::Number) where {T<:Number} = x - T(opt.rate) * Δx

function apply_opt(opt::GradientDecent, x::AbstractArray{T}, Δx::AbstractArray{<:Number}) where {T<:Number}
    x .- T(opt.rate) .* Δx
end


# Let's test it:

optimizer = GradientDecent(1)

apply_opt(optimizer, m, grad_model) isa typeof(m)


# ### Training the model

m_trained = deepcopy(m)

learn_schedule = [
    (batchsize = 1000, optimizer = GradientDecent(0.1), epochs = 1),
    (batchsize = 5000, optimizer = GradientDecent(0.05), epochs = 1),
    (batchsize = 50000, optimizer = GradientDecent(0.025), epochs = 1),
]

loss_history = Float64[]
loss_ninputs = Int[]

ProgressMeter.ijulia_behavior(:clear)
let m = m_trained
    n_input_total = size(L_train, 2) * sum([p.epochs for p in learn_schedule])
    n_done = 0
    p = ProgressMeter.Progress(n_input_total, 0.1, "Training...")
    
    for lern_params in learn_schedule
        batchsize = lern_params.batchsize
        optimizer = lern_params.optimizer
        for epoch in 1:lern_params.epochs
            shuffled_idxs = shuffle(rng, axes(L_train, 2))
            partitions = partition(adapt(ArrayType, shuffled_idxs), batchsize)
    
            for idxs in partitions
                L = L_train[:, idxs]
                X = X_train[:, idxs]
    
                f_loss = f_loglike_loss(binary_xentropy, L)
                f_model_loss = f_loss ∘ m
                
                loss_current_batch = f_model_loss(X)
                grad_model_loss = pullback(one(Float32), f_model_loss, X)
                grad_model = grad_model_loss[1].inner
    
                m = apply_opt(optimizer, m, grad_model)
    
                push!(loss_history, loss_current_batch)
                push!(loss_ninputs, n_done)
    
                n_done += length(idxs)
                ProgressMeter.update!(p, n_done; showvalues = [(:batchsize, batchsize), (:optimizer, optimizer), (:loss_current_batch, loss_current_batch),])
            end
        end
    end
    ProgressMeter.finish!(p)
        
    global m_trained = m
end

plot(loss_ninputs, loss_history)


# ### Analysis of trained model

# To run over large dataset we need to use batches that fit into GPU memory:

function batched_eval(m, X::AbstractMatrix{<:Real}; batchsize::Integer = 50000)
    Y = similar(X, size(m(X[:,1]))..., size(X, 2))
    idxs = axes(X, 2)
    partitions = partition(idxs, batchsize)
    batch_idxs = first(partitions)
    for batch_idxs in partitions
        view(Y, :, batch_idxs) .= m(view(X, :, batch_idxs))
    end
    return Y
end


# Let's compute some common metrics and generate a nice plot:

Y_train_v = Array(vec(batched_eval(m_trained, X_train)))

Y_test_v = Array(vec(batched_eval(m_trained, X_test)))

L_all_v = Array(vec(L_all))
L_train_v = Array(vec(L_train))
L_test_v = Array(vec(L_test))


pred_sortidxs = sortperm(Y_test_v)
pred_sorted = Y_test_v[pred_sortidxs]
truth_sorted = L_test_v[pred_sortidxs]

n_true_over_pred = reverse(cumsum(reverse(truth_sorted)))
n_false_over_pred = reverse(cumsum(reverse(.!(truth_sorted))))
n_total = length(truth_sorted)
n_true = first(n_true_over_pred)
n_false = first(n_false_over_pred)

thresholds = 0.001:0.001:0.999
found_thresh_ixds = searchsortedlast.(Ref(pred_sorted), thresholds)
get_or_0(A, i::Integer) = get(A, i, zero(eltype(A)))
TPR = [get_or_0(n_true_over_pred, i) / n_true for i in found_thresh_ixds]
FPR = [get_or_0(n_false_over_pred, i) / n_false for i in found_thresh_ixds]

plot(
    plot(loss_ninputs, loss_history, label = "Training evolution", xlabel = "n_inputs", ylabel = "loss"),
    begin
        plot(xlabel = "model output")
        stephist!(Y_test_v[findall(L_test_v)], nbins = 100, normalize = true, label = "Test pred. for true pos.", lw = 3),
        stephist!(Y_test_v[findall(.! L_test_v)], nbins = 100, normalize = true, label = "Test pred. for true neg.", lw = 3)
        stephist!(Y_train_v[findall(L_train_v)], nbins = 100, normalize = true, label = "Training pred. for true pos.", lw = 2)
        stephist!(Y_train_v[findall(.! L_train_v)], nbins = 100, normalize = true, label = "Training pred. for true neg.", lw = 2)
    end,
    begin
        plot(thresholds, TPR, label = "TPR", color = :green, xlabel = "treshold", lw = 2)
        plot!(thresholds, FPR, label = "FPR", color = :red, lw = 2)
    end,
    plot(FPR, TPR, label = "ROC", xlabel = "FPR", ylabel = "TPR", lw = 2),
)

# To save the plot, use

#savefig("result.png")