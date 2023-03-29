# This file is licensed under the MIT License (MIT).

using BenchmarkTools, Test

using CUDA
GPUArray = CuArray

#=
using Metal
GPUArray = MtlArray
=#

gpu_features = adapt(GPUArray, features);
gpu_labels = adapt(GPUArray, labels);

gpu_model = adapt(GPUArray, model)
typeof(gpu_model)

idxs = 1:50000
X = features[:, idxs]
L = labels[:, idxs]
gpu_X = gpu_features[:, idxs]
gpu_L = gpu_labels[:, idxs]

δY = model(X)
gpu_δY = gpu_model(gpu_X)

typeof(@inferred pullback(δY, model, X))
typeof(@inferred pullback(gpu_δY, gpu_model, gpu_X))

#=
@benchmark sum($model($X))
@benchmark sum($gpu_model($gpu_X))
=#

#=
@benchmark sum(pullback($δY, $model, $X)[2])
@benchmark sum(pullback($gpu_δY, $gpu_model, $gpu_X)[2])
=#


f_loss = f_loglike_loss(binary_xentropy, L)
gpu_f_loss = f_loglike_loss(binary_xentropy, gpu_L)

@inferred (f_loss ∘ model)(X)
@inferred (gpu_f_loss ∘ gpu_model)(gpu_X)

typeof(@inferred pullback(one(Float32), f_loss ∘ model, X))
typeof(@inferred pullback(one(Float32), gpu_f_loss ∘ gpu_model, gpu_X))

# Should be about 200 ms:
@benchmark pullback(one(Float32), $f_loss ∘ $model, $X)
# Should be about 20 ms:
@benchmark pullback(one(Float32), $gpu_f_loss ∘ $gpu_model, $gpu_X)


#=
import Zygote
Zygote.gradient(gpu_f_loss ∘ Base.Fix2((f,x) -> f(x), gpu_X), gpu_model)
@benchmark Zygote.gradient($gpu_f_loss ∘ Base.Fix2((f,x) -> f(x), view($gpu_features, :, idxs)), $gpu_model)
=#
