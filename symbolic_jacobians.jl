# This software is licensed under the MIT "Expat" License.

using Symbolics

@variables J_f[1:1, 1:3] J_g[1:3, 1:4] J_h[1:4, 1:5]

collect(J_f * J_g * J_h)

collect(J_g * J_h)

# A dummy ML model:

@variables θ[1:4]
dummy_loss = let θ = collect(θ), x = rand(4), y = rand(4)
    -sum( ((θ .+ x) .- y) .^ 2)
end

# Its gradient in respect to the parameter vector θ:

[Symbolics.derivative(dummy_loss, θ[i]) for i in 1:4]

# If we treat the loss as a one-element vector, we can compute the gradient
# as a Jacobian - only in the Jacobian view, it's a row vector:

Symbolics.jacobian([dummy_loss], θ)


# Now let's build a more serious categorical model:

logistic(x::Real) = inv(exp(-x) + one(x))

softmax(X::VecOrMat{<:Real}) = exp.(X) ./ sum(exp.(X), dims = 1)

categorical_logpdf(p::AbstractVector{<:Real}, obs::Integer) = log(p[obs])


l1 = let
    @variables A[1:4,1:5] b[1:4]
    x -> logistic.(collect(A) * collect(x) .+ collect(b))
end

l2 = let
    @variables A[1:3,1:4] b[1:3]
    x -> logistic.(collect(A) * collect(x) .+ collect(b))
end

f_model = softmax ∘ l2 ∘ l1


# Some dummy input and training output data (single sample):

x = rand(5)
y = 1

# Now we can get the loss as a symbolic expression:

loss = - categorical_logpdf(f_model(x), y)

# And we can get the Jacobian of the loss in respect to the bias of layer 1:
J_loss_l1_b = Symbolics.jacobian([loss], l1.b)


# Since the loss is a scalar, the Jacobian has a single row, it equals the
# transposed gradient vector:

size(J_loss_l1_b)

# So we can get the gradient of the loss in respect to any parameter array,
# e.g. in respect to the weights of layer 2:

J_loss_l2_A = Symbolics.jacobian([loss], l2.A)

# The math gets pretty long, though ... and that's for just one sample!
