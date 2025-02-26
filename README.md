# Machine learning from scratch in Julia

This is a "Machine-learning from scratch" tutorial in Julia. It demonstrates
how to implement a simple artificial neural network with automatic
differentiation and a simple gradient descent optimizer, using only the
Julia standard library and a (very) few data structure packages.

This tutorial uses the
[UCI ML SUSY Data Set](https://archive.ics.uci.edu/ml/datasets/SUSY), a
binary clasification dataset with 5 million events and 18 features.

Install and configure Julia: If you're new to Julia we recommend you follow
the [instuctions linked here](https://github.com/oschulz/julia-setup).

Now open a [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/)
and go through the following steps to

* Install all required Julia packages required for this tutorial

* Generate a Jupyter notebook version "ml_from_scratch.ipynb" of the tutorial.

```
julia> cd("PATH/TO/YOUR/DOWNLOAD/OF/julia-ml-from-scratch")
# Press "]" key to enter the Pkg console, then
(@v1.11) pkg> activate .
(julia-ml-from-scratch) pkg> instantiate
# Press backspace (or <ctrl-C>) to exit the Pkg console, then
julia> include("generate_notebook.jl")
```

If you prefer working with Julia scripts instead of Jupyter notebooks, simply
run

```
julia> include("ml_from_scratch.jl")
```

on the Julia REPL to run the whole tutorial in one go, or run sections of
"ml_from_scratch.jl" manually, e.g. in Visual Studio Code.
