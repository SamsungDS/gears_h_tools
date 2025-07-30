# GEARS H Tools

`gears_h_tools` is a helper package for `gears_h`.
Currently, the main functionality is to convert `gpaw` LCAO calculation outputs to the input format for `gears_h` training.
`gears_h_tools` can also be used to compute the $S$-matrix for a structure, which is helpful for inferred Hamiltonians, as you will need the $S$-matrix to compute the eigenvalues, among many other properties.

## Installation

`gears_h_tools` requires `gpaw` to function.
As discussed in [the next section](#speeding-up-conversion), `gpaw>=25.7.0` is recommended.

The easiest way to install `gpaw` such that it has `scalapack` and `elpa` to accelerate calculations` is with `spack`.
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
