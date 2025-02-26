# This software is licensed under the MIT "Expat" License.

module DownloadSUSYDataset

import Downloads
import Tables, CSV, HDF5


function download_dataset()
    datadir = get(ENV, "MLFS_DATADIR", @__DIR__)

    csv_filename = joinpath(datadir, "SUSY.csv.gz")
    hdf5_filename = joinpath(datadir, "SUSY.hdf5")

    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"

    if !isfile(hdf5_filename)
        @info "\"$hdf5_filename\" doesn't exist, generating it."

        if !isfile(csv_filename)
            @info "\"$csv_filename\" doesn't exist, downloading it from \"$dataset_url\"."
            Downloads.download(dataset_url, csv_filename)
        end

        @info "Converting CSV to HDF5."

        @time A = Tables.matrix(CSV.File(csv_filename; header=false, buffer_in_memory=true, types=Float32))

        labels = Int32.(A[:, begin])
        features = A[:, begin+1:end]

        HDF5.h5open(hdf5_filename, "w") do output
            output["labels"] = labels
            output["features"] = features
        end
    else
        @info "Found dataset at \"$hdf5_filename\"."
    end

    return hdf5_filename
end

end # module


DownloadSUSYDataset.download_dataset()
