# This script is licensed under the MIT License (MIT).

using Pkg
Pkg.activate(@__DIR__)
#Need to run this only once:
Pkg.instantiate()

import Literate

Literate.notebook("hep_ml_from_scratch.jl", ".", execute = false, name = "hep_ml_from_scratch", credit = true)
