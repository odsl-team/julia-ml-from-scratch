# HEP ML from scratch in Julia

This is a Machine-learning from scratch" tutorial in Julia that uses a high energy physics (HEP) ML test dataset.

Note: Please use Julia v1.9 to run this tutorial!

The main tutorial file is "hep_ml_from_scratch.jl".

First, install the package [IJulia](https://github.com/JuliaLang/IJulia.jl)
packag in your Julia default project environment:

```shell
julia 'using Pkg; Pkg.add("IJulia")'
```

Then run

```shell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

to ensure all required Julia packages are installed properly.

Afterwards, run

```shell
julia --project=. generate_notebook.jl
```

to generate a Jupyter notebook version "hep_ml_from_scratch.ipynb" of the tutorial.

Alternatively, enter the Julia package management console by pressing "]" and do

```julia
julia> ]
(@v1.9) pkg> add IJulia
(@v1.9) pkg> activate .
(julia-hepml-from-scratch) pkg> instantiate
# Backspace key
julia> include("generate_notebook.jl")
```
