# HEP ML from scratch in Julia

This is a Machine-learning from scratch" tutorial in Julia that uses a high energy physics (HEP) ML test dataset.

Note: Please use Julia v1.9 to run this tutorial!

The main tutorial file is "hep_ml_from_scratch.jl".

Run

```shell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

to ensure all required Julia packages are installed properly.

Run

```shell
julia --project=. generate_notebook.jl
```

to generate a Jupyter notebook version "hep_ml_from_scratch.ipynb" of the tutorial.

Alternatively, enter the Julia package management console by pressing "]" and do

```julia
julia> ]
(@v1.9) pkg> activate .
(julia-hepml-from-scratch) pkg> instantiate
```
