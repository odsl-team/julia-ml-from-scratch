import Tables, CSV, HDF5

input_filename = ARGS[1]

if !isfile(input_filename)
    error("$input_filename doesn't exist")
end

output_filename = replace(input_filename, r"\.csv\.gz$" => "")*".hdf5"

@time A = Tables.matrix(CSV.File(input_filename; header=false, buffer_in_memory=true, types=Float32))

tags = Int32.(A[:, begin])
features = A[:, begin+1:end]

HDF5.h5open(output_filename, "w") do output
    output["tags"] = tags
    output["features"] = features
end
