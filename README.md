# HEP ML from scratch in Julia

This is a Machine-learning from scratch" tutorial in Julia that uses a high
energy physics (HEP) ML test dataset.

The tutorial uses the
[UCI ML SUSY Data Set](https://archive.ics.uci.edu/ml/datasets/SUSY), a
binary clasification dataset with 5 million events and 18 features.

You please download and install
[Julia v1.9](https://julialang.org/downloads/#upcoming_release) to run this
tutorial. On Windows (only!), please make sure you have the
[Visual C++ redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)
installed.


You should also install [Visual Studio Code](https://code.visualstudio.com/download)
and/or working Jupyter installation.
[JupyterLab Desktop](https://github.com/jupyterlab/jupyterlab-desktop/releases)
is easy to install (but a full Anaconda or custom Python
installation with Jupyter will work too, of course).
Also install the *pre-release* version of the
[VS-Code Julia extension](https://code.visualstudio.com/docs/languages/julia).

Now open a [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/)
and go through the following steps to

* Install all required Julia packages required for this tutorial

* Install an [IJulia](https://github.com/JuliaLang/IJulia.jl) Jupyter Julia
kernel tied to this tutorial.

* Generate a Jupyter notebook version "hep_ml_from_scratch.ipynb" of the tutorial.

```
julia> cd("PATH/TO/YOUR/DOWNLOAD/OF/julia-hepml-from-scratch")
# Press "]" key to enter the Pkg console, then
(@v1.9) pkg> activate .
(julia-hepml-from-scratch) pkg> instantiate
# Press backspace (or <ctrl-C>) to exit the Pkg console, then
julia> include("generate_notebook.jl")
```
