# This software is licensed under the MIT "Expat" License.

import Pkg

# Activate tutorial environment and generate Jupyter notebook

import Pkg
if basename(dirname(Pkg.project().path)) != "julia-ml-from-scratch"
    Pkg.activate(@__DIR__)
end
#Pkg.instantiate() # Need to run this only once

import Literate

Literate.notebook("ml_from_scratch.jl", ".", execute = false, name = "ml_from_scratch", credit = true)
