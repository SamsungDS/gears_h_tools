# GEARS H Tools

`gears_h_tools` is a helper package for [`gears_h`](https://github.com/SamsungDS/gears_h/tree/main).
Currently, the main functionality is to convert `gpaw` LCAO calculation outputs to the input format for `gears_h` training.
`gears_h_tools` can also be used to compute the $S$-matrix for a structure, which is helpful for inferred Hamiltonians, as you will need the $S$-matrix to compute the eigenvalues, among many other properties.

## Installation

`gears_h_tools` requires `gpaw` to function.
As discussed in [the next section](#speeding-up-conversion), `gpaw>=25.7.0` is recommended.

> **WARNING**: there is currently a bug in the released version of `gpaw` 25.7.0 that breaks the extraction of LCAO Hamiltonians.
> A bug fix is pending (see [here](https://gitlab.com/gpaw/gpaw/-/merge_requests/2836) for details), but until it has been merged and a new release containing it is available, you will either need to manually add the fix yourself or use an older version of `gpaw`. If you choose the latter approach, simply change the version number below.

The easiest way to install `gpaw` such that it has `scalapack` and `elpa` to accelerate calculations is with `spack`.
First, follow spack's installation instructions [here](https://github.com/spack/spack?tab=readme-ov-file#installation).

Then, run the following commands:
```shell
spack env create gears_h_tools
spacktivate -p gears_h_tools
spack add py-pip py-gpaw@25.7.0
spack concretize
spack install
```

This should successfully install `gpaw` v25.7.0. 
If later versions of `gpaw` are available, simply adjust the version number.

Next, we need to install `gears_h_tools` into the same environment:

```shell
cd path/to/gears_h_tools
pip install .
```

Whenever you need to use `gears_h_tools`, simply set up your `spack` environment as you did in the installation instructions and then run `spacktivate -p gears_h_tools`.

## Speeding up conversion

We recommend using `gpaw>=25.7.0` for converting your output files to `gears_h` training data because it includes [this commit](https://gitlab.com/gpaw/gpaw/-/merge_requests/2817), which greatly accelerates the process.
If you are on an older version of `gpaw` for whatever reason, you can backport this commit to your installation.

## Examples

The primary user-facing function is `gears_h_tools.prepare_data.prepare_gpaw_gears_h_snapshot`.
Here, we will present a few examples of applying this function.

### Converting gpaw calculations to training/validation data

The following script will (once the requisite function arguments have been filled in) read in the gpaw output file with path `gpwfilename` and write it to `"path/to/output/directory"`.
It will write out all the on-diagonal Hamiltonian blocks and off-diagonal Hamiltonian blocks for atoms within `CUTOFF` angstroms apart.
Hamiltonian blocks with a maximum absolute value smaller than `1e-4` eV will not be written out, to prevent unnecessary data from being written out.
This threshold is controllable using the `block_write_threshold` argument, if you wish to change it.

```python
from pathlib import Path

from gears_h_tools.prepare_data import prepare_gpaw_slh_snapshot

prepare_gpaw_slh_snapshot(directory = Path("path/to/output/directory"),
                          gpwfilename = gpwfilename, 
                          cutoff = CUTOFF)
```

### Parallelizing conversions

In general, you will have many training structures to convert.
Here is a sample script for converting many structures in parallel:

```python
from multiprocessing import Pool
from pathlib import Path

from gears_h_tools.prepare_data import prepare_gpaw_slh_snapshot
from tqdm import tqdm

def prepare_ith_datapoint(gpwfilename):
    prepare_gpaw_slh_snapshot(
        Path("processed_data/") / gpwfilename.parent.name,
        gpwfilename=gpwfilename, 
        cutoff=CUTOFF
    )

with Pool(24) as p:
    gpwfilelist = list(Path("path/to/gpaw/calculation/root/directory").glob("**/*.gpw"))
    for i in tqdm(
        p.imap_unordered(func=prepare_ith_datapoint, iterable=gpwfilelist),
        total=len(gpwfilelist),
    ):
        continue
```

This script assumes the following directory structure: in `path/to/gpaw/calculation/root/directory` you have distinct directories with `gpw` files inside of them.
For each `gpw` file, it creates a directory in `processed_data` with the name of the folder containing the `gpw` file, and outputs the converted data there.
The `processed_data` directory is the one you will point `gears_h` to.
This script runs 24 conversions at a time, and should be run with this command: `OMP_NUM_THREADS=1 python convert.py`

Adjust this example script as needed for your system and use case.

### Calculating $S$-matrices

When diagonalizing inferred Hamiltonians, you will need to compute the $S$-matrices for the structures.
Fortunately, these are relatively light calculations, so you should be able to do them even for large systems.
However, we strongly recommend you follow [the speedup advice above](#speeding-up-conversion) for large structures.

Calculating $S$-matrices proceeds very similarly to the procedures above, with the Hamiltonian conversion disabled.
Since there won't be a pre-existing `gpaw` output file to read in calculation parameters from, we also need to provide those.
This example script implements a parallel $S$-matrix calculation:

```python
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm
from gears_h_tools.prepare_data import prepare_gpaw_slh_snapshot

overlap_calc_parameters = {"mode": "lcao",
                           "basis" : {"default": "szp(dzp)"},
                           "xc" : "PBE",
                           "kpts" : [1, 1, 1],
                           "parallel" : {"sl_auto": True,
                                         "use_elpa": True,
                                         "augment_grids": True}
                          }

def prepare_ith_datapoint(gpwfilename):
    prepare_gpaw_slh_snapshot(
        Path(f"processed_data/") / gpwfilename.parent.name,
        gpwfilename=gpwfilename,
        cutoff=CUTOFF,
        overlap_only = True,
        overlap_calc_parameters = overlap_calc_parameters
    )


with Pool(8) as p:
    root_dir = Path("path/to/structure/root/dir")
    gpws = sorted(root_dir.glob("**/*.extxyz"))
    for i in tqdm(
        p.imap_unordered(func=prepare_ith_datapoint,
                         iterable=gpws),
        total=len(gpws),
    ):
        continue

```

This script finds all `extxyz` structure files in directories in `path/to/structure/root/dir`, and outputs the calculated $S$-matrices in directories with the same name as the directories containing the structure files but in `processed_data` instead.
The major differences from above are:
1. We set `overlap_only = True` in `prepare_gpaw_slh_snapshot`.
2. We also pass `overlap_calc_parameters` to `prepare_gpaw_slh_snapshot`.
3. We pass the structure files in `gpwfilename`, rather than actual `gpw` files.

The $S$-matrix calculation can benefit from threading.
Make sure, however, that `OMP_NUM_THREADS*n_procs` (`n_procs` is the argument of `Pool`) does not exceed the number of cores on your system.

## Authors

GEARS H Tools was designed and built by
- Anubhab Haldar
- Ali K. Hamze

under the supervision of Yongwoo Shin.

## References

If you use this code, please cite our paper:

```bibtex
@online{haldarGEARSAccurateMachinelearned2025,
  title = {{{GEARS H}}: {{Accurate}} Machine-Learned {{Hamiltonians}} for next-Generation Device-Scale Modeling},
  shorttitle = {{{GEARS H}}},
  author = {Haldar, Anubhab and Hamze, Ali K. and Sivadas, Nikhil and Shin, Yongwoo},
  date = {2025-06-12},
  eprint = {2506.10298},
  eprinttype = {arXiv},
  eprintclass = {cond-mat},
  doi = {10.48550/arXiv.2506.10298},
  url = {http://arxiv.org/abs/2506.10298},
  urldate = {2025-06-13},
  abstract = {We introduce GEARS H, a state-of-the-art machine-learning Hamiltonian framework for large-scale electronic structure simulations. Using GEARS H, we present a statistical analysis of the hole concentration induced in defective \$\textbackslash mathrm\{WSe\}\_2\$ interfaced with Ni-doped amorphous \$\textbackslash mathrm\{HfO\}\_2\$ as a function of the Ni doping rate, system density, and Se vacancy rate in 72 systems ranging from 3326 to 4160 atoms-a quantity and scale of interface electronic structure calculation beyond the reach of conventional density functional theory codes and other machine-learning-based methods. We further demonstrate the versatility of our architecture by training models for a molecular system, 2D materials with and without defects, solid solution crystals, and bulk amorphous systems with covalent and ionic bonds. The mean absolute error of the inferred Hamiltonian matrix elements from the validation set is below 2.4 meV for all of these models. GEARS H outperforms other proposed machine-learning Hamiltonian frameworks, and our results indicate that machine-learning Hamiltonian methods, starting with GEARS H, are now production-ready techniques for DFT-accuracy device-scale simulation.},
  pubstate = {prepublished},
  keywords = {Condensed Matter - Materials Science,Physics - Computational Physics},
}
```
