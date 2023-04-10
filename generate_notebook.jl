# This script is licensed under the MIT License (MIT).

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate() # Need to run this only once

import IJulia

kernelname = "Julia ML from Scratch"
kernelspec = IJulia.installkernel(kernelname, "--project=$(dirname(Pkg.project().path))")

import Literate

function set_kernelspec!(content)
    content["metadata"]["kernelspec"]["display_name"] = "$kernelname $VERSION"
    content["metadata"]["kernelspec"]["name"] = basename(kernelspec)
    return content
end

Literate.notebook("hep_ml_from_scratch.jl", ".", execute = false, name = "hep_ml_from_scratch", credit = true, postprocess = set_kernelspec!)
