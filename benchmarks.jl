# This file is licensed under the MIT License (MIT).

using BenchmarkTools
import Zygote


using CUDA
GPUArray = CuArray

#=
using Metal
GPUArray = MtlArray
=#


gpu_model = adapt(GPUArray, model)
typeof(gpu_model)
gpu_dY = adapt(GPUArray, dY)
gpu_X = adapt(GPUArray, X)
gpu_model(gpu_X)
typeof(pullback(gpu_dY, gpu_model, gpu_X))

#@benchmark $model($X)
#@benchmark $gpu_model($gpu_X)

#@benchmark pullback($dY, $model, $X)
#@benchmark pullback($gpu_dY, $gpu_model, $gpu_X)

gpu_features = adapt(GPUArray, features);
gpu_labels = adapt(GPUArray, labels);

idxs = 1:50000

Y = model(view(features, :, idxs));
gpu_Y = gpu_model(view(gpu_features, :, idxs));

#@benchmark $model(view($features, :, idxs))
#@benchmark $gpu_model(view($gpu_features, :, idxs))

#@benchmark sum(pullback($Y, $model, view($features, :, idxs))[2])
#@benchmark sum(pullback($gpu_Y, $gpu_model, view($gpu_features, :, idxs))[2])

f_loss = f_xentroy_loss(view(labels, idxs))
gpu_f_loss = f_xentroy_loss(view(gpu_labels, idxs))

(f_loss ∘ model)(view(features, :, idxs))
(gpu_f_loss ∘ gpu_model)(view(gpu_features, :, idxs))

typeof(pullback(one(Float32), f_loss ∘ model, view(features, :, idxs)))
typeof(pullback(one(Float32), gpu_f_loss ∘ gpu_model, view(gpu_features, :, idxs)))

# Should be about 200 ms:
@benchmark pullback(one(Float32), $f_loss ∘ $model, view($features, :, idxs))
# Should be about 20 ms:
@benchmark pullback(one(Float32), $gpu_f_loss ∘ $gpu_model, view($gpu_features, :, idxs))

Zygote.gradient(gpu_f_loss ∘ Base.Fix2((f,x) -> f(x), view(gpu_features, :, idxs)), gpu_model)
@benchmark Zygote.gradient($gpu_f_loss ∘ Base.Fix2((f,x) -> f(x), view($gpu_features, :, idxs)), $gpu_model)
