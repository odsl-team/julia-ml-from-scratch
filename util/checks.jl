# This file is licensed under the MIT License (MIT).

using Test

include("approx_cmp.jl")

idxs = 1:50000
L = labels[:, idxs]
X = features[:, idxs]
f_loss = f_loglike_loss(binary_xentropy, L)

@test test_pullback(model, X)
@test test_pullback(f_loss ∘ model, X)

@inferred pullback(one(eltype(X)), f_loss ∘ model, X)
