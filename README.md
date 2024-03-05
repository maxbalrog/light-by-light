# light-by-light

This package provides helpful functions to run gridscan simulations and optuna optimizations using Vacuum Emission solver [1]. There are also routines to create corresponding bash scripts on Draco cluster. 

- usage.ipynb illustrates typical gridscan and optuna optimization scenario
- create_bash.ipynb shows the routines to create bash scripts for Draco

This package was used to obtain results for our paper [2].

## Potential improvements -> new repo
1. Since it is not easy to get access to optuna surrogate model and posterior distribution of TPE sampler, we plan to switch to a more versatile Bayesian library: BoTorch and Ax.
2. With the help of high-level schedulers it might be possible to make more efficient and better structured implementation of grid scans and optimizations.


## Resources
[1] - A. Blinne, et al. "All-optical signatures of quantum vacuum nonlinearities in generic laser fields." Physical Review D 99.1 (2019): 016006.

[2] - In progress...
