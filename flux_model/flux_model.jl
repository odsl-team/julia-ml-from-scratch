# This file is licensed under the MIT License (MIT).

import Flux, Optimisers, MLUtils
import Plots, ProgressMeter
import HDF5

datadir = get(ENV, "MLFS_DATADIR", dirname(@__DIR__))

input = HDF5.h5open(joinpath(datadir, "SUSY.hdf5"))
features = copy(transpose(read(input["features"])))
labels = Bool.(transpose(read(input["labels"])))


idxs_total = eachindex(labels)
n_total = length(idxs_total)
n_train = round(Int, 0.7 * n_total)
idxs_train = 1:n_train
idxs_test = n_train+1:n_total

L_train = Flux.gpu(labels[:, idxs_train])
L_test = Flux.gpu(labels[:, idxs_test])
X_train = Flux.gpu(features[: ,idxs_train])
X_test = Flux.gpu(features[:, idxs_test])


flux_model = Flux.Chain(
    Flux.Dense(18, 128, Flux.relu),
    Flux.Dense(128, 128, Flux.relu),
    Flux.Dense(128, 1, Flux.sigmoid)
) |> Flux.gpu


Flux.binarycrossentropy(flux_model(X_train[:, 1:10000]), L_train[:,1:10000])


optim = Flux.setup(Optimisers.Adam(), flux_model)
n_epochs = 3
batchsize = 50000

dataloader = MLUtils.DataLoader((X_train, L_train), batchsize=batchsize, shuffle=true)

loss_history = zeros(0)
p = ProgressMeter.Progress(n_epochs * length(dataloader), dt=0.1, desc="Training...")
for epoch in 1:n_epochs
    for (x, y) in dataloader
        loss_train, grads = Flux.withgradient(flux_model) do m
            # Evaluate flux_model and loss inside gradient context:
            y_hat = m(x)
            Flux.binarycrossentropy(y_hat, y)
        end
        push!(loss_history, loss_train)
        ProgressMeter.next!(p; showvalues = [(:loss_train, loss_train),#= (:loss_test, loss_test)=#])
        Flux.update!(optim, flux_model, grads[1])
    end
end
ProgressMeter.finish!(p)

Plots.plot(loss_history)
