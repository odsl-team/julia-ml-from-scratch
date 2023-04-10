# *This software is licensed under the MIT "Expat" License (MIT).*

# ## ML from scratch in Julia
#
# April 2023, Oliver Schulz <oschulz@mpp.mpg.de>

# ### Julia project environment

# Ensure that the right Julia project environment is active:

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate() # Need to run this only once
basename(dirname(Pkg.project().path))


# ### Dependencies

# Julia standard libary functionality:

using LinearAlgebra, Statistics, Random

using Base: Fix1
using Base.Broadcast: BroadcastFunction
using Base.Iterators: partition

# Some data structure packages that we'll need:

using Adapt, StructArrays, ConstructionBase

# Plotting and I/O packages:

using Plots, ProgressMeter
using HDF5

# Run on CPU by default:

ArrayType = Array

# To use NVIDIA CUDA, uncomment and run:

#Pkg.add("CUDA") # Need to run this only once
#using CUDA
#ArrayType = CuArray

# To use AMD ROCm (AMDGPU.jl still maturing, do not expect competitive
# performance), uncomment and run:

#Pkg.add("AMDGPU") # Need to run this only once
#using AMDGPU
#ArrayType = ROCArray

# To try using Intel oneAPI (oneAPI.jl is not very mature yet, do not expect
# acceptable performance and do expect memory management issues), uncomment
# and run:

#Pkg.add("oneAPI") # Need to run this only once
#using oneAPI
#ArrayType = oneArray

# To try using Apple Metal (Metal.jl is not very mature yet, do not expect
# acceptable performance and do expect memory management issues), uncomment
# and run:

#Pkg.add("METAL") # Need to run this only once
#using Metal
#ArrayType = MtlArray


# ### Dataset

# We'll use the SUSY Data Set from the
# [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SUSY).
#
# This will download the original "SUSY.csv.gz" and convert it to "SUSY.hdf5",
# need to run this only once:

include("download_dataset.jl")

# Open "SUSY.hdf5" and read features and labels:

input = h5open("SUSY.hdf5")
features = copy(transpose(read(input["features"])))
labels = Bool.(transpose(read(input["labels"])))


# ### A simple automatic differentiation system

struct NoTangent end
const δ∅ = NoTangent()
Base.sum(::AbstractArray{<:NoTangent}) = δ∅

pullback(δy, ::typeof(+), a, b) = (δ∅, δy, δy)

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


unbroadcast(δY::NoTangent, size_X::Tuple{}) = δY
unbroadcast(δY::NoTangent, size_X::Dims) = δY

unbroadcast(δY, size_X::Tuple{}) = sum(δY)

function unbroadcast(δY::AbstractArray{<:Any,N}, size_X::Dims{M}) where {N,M}
    if size(δY) == size_X
        return δY
    else
        #Trick to get type-stable dims to sum over, sum ignores non-exsisting dims:
        summing_dims = ntuple(d -> get(size_X, d, 1) == 1 ? d : N+100, N)
        return reshape(sum(δY, dims = summing_dims), size_X...)
    end
end

_bcf_pullback_kernel(δy_ntpl1, args...) = pullback(only(δy_ntpl1), args...)

function pullback(δY, bf::BroadcastFunction, Xs...)
    #Wrap δY in StructArray with eltype NTuple{1} to make broadcast return a
    #StructArray. Easy GPU-compatible way to separate tangent components:
    δY_ntpl1_sa = StructArray((δY,))
    δ_f_Xs_sa = broadcast(_bcf_pullback_kernel, δY_ntpl1_sa, bf.f, Xs...)
    δ_f_Xs = StructArrays.components(δ_f_Xs_sa)

    δ_f = sum(first(δ_f_Xs))
    δ_f_Xs = map(unbroadcast, Base.tail(δ_f_Xs), map(size, Xs))

    return (δ_f, δ_f_Xs...)
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
    randn!(rng, A)
    A .= gain .* scale .* A
    return A
end

# Function to construct a linear neural network layer that implements the function
# $$A * x + b$$

function linear_layer(rng::AbstractRNG, n_in::Integer, n_out::Integer, f_init! = glorot_uniform!, ::Type{T} = Float32) where {T<:Real}
    A = Matrix{T}(undef, n_out, n_in)
    f_init!(rng, A)
    b = similar(A, n_out)
    fill!(b, zero(T))
    return Fix1(.+, b) ∘ Fix1(*, A)
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
    linear_layer(rng, 18, 128),
    BroadcastFunction(relu),
    linear_layer(rng, 128, 128),
    BroadcastFunction(relu),
    linear_layer(rng, 128, 1),
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

# The binary cross-entroy is the negative log-likelihood of a Bernoulli
# distribution (uncomment to verify):

#Pkg.add("Distributions") # Need to run this only once
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

#Recursive views can cause trouble on some GPU backends, so copy SubArrays:

maybe_copy(A::AbstractArray) = A
maybe_copy(A::SubArray) = copy(A)

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
apply_opt(opt::AbstractOptimizer, x::NTuple{N}, δx::NTuple{N}) where N = map((xi, dxi) -> apply_opt(opt, xi, dxi), x, δx)
apply_opt(opt::AbstractOptimizer, x::NamedTuple{names}, δx::NamedTuple{names}) where names = map((xi, dxi) -> apply_opt(opt, xi, dxi), x, δx)
apply_opt(opt::AbstractOptimizer, x::T, δx) where T = constructorof(T)(values(apply_opt(opt, getfields(x), δx))...)


# A simple gradient decent optimizer with fixed learning rate:

struct GradientDecent{T} <: AbstractOptimizer
    rate::T
end

apply_opt(opt::GradientDecent, x::T, δx::Number) where {T<:Number} = x - T(opt.rate) * δx

function apply_opt(opt::GradientDecent, x::AbstractArray{T}, δx::AbstractArray{<:Number}) where {T<:Number}
    x .- T(opt.rate) .* δx
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

in_vscode_notebook = haskey(ENV, "VSCODE_CWD")
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
                if !in_vscode_notebook
                    #ProgessMeter output doesn't work well in VSCode notebooks yet.
                    ProgressMeter.update!(p, n_done; showvalues = [(:batchsize, batchsize), (:optimizer, optimizer), (:loss_current_batch, loss_current_batch),])
                end
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
