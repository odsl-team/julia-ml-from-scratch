# This software is licensed under the MIT "Expat" License.

import Pkg

# Ensure IJulia is installed in default Julia environment

Pkg.activate() # Activate default environment
Pkg.instantiate() # Need to run this only once

if !("IJulia" in keys(Pkg.project().dependencies))
    @info "Installing IJulia into default Julia environment \"$(Pkg.project().path)\""
    Pkg.add("IJulia")
    Pkg.build("IJulia")
    # Should not be necessary:
    # IJulia.installkernel("Julia", "--project=@.")
end
import IJulia


# Activate tutorial environment and generate Jupyter notebook

Pkg.activate(@__DIR__)
Pkg.instantiate() # Need to run this only once

import Literate

Literate.notebook("ml_from_scratch.jl", ".", execute = false, name = "ml_from_scratch", credit = true)
