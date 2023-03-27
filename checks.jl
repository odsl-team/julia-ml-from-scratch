# This file is licensed under the MIT License (MIT).

include("approx_cmp.jl")

idxs = 1:50000
f_loss = f_xentroy_loss(view(labels, idxs))
x = view(features, :, idxs)
@assert test_pullback(model, x)
@assert test_pullback(f_loss ∘ model, x)
