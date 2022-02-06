# Download petsc
```
$ git clone -b release https://gitlab.com/petsc/petsc.git petsc
```

# Install
```
$ cd petsc
```

```
$ ./configure --with-debugging=0 --with-debugging=0 --download-p4est=1 --with-zlib=1 --with-fortran-bindings=0 --download-parmetis=1 --download-metis=1 -download-hdf5=1 --download-hypre=1 --download-scalapack --download-superlu --download-superlu_dist --download-mumps --with-petsc4py
```

```
$ make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-linux-c-opt all
```

```
$ python3 -m pip install --user -e .
```


# Run mfg_parallel_petsc_pyccel.py

```
$ mpirun -n 2 python mfg_parallel_petsc_pyccel.py -pc_factor_mat_solver_type mumps
```

## Optional commands:

- `-snes_linesearch_type l2` L2norm for linear search (used when the initial guess is activated)
- `-snes_lag_jacobian -2` compute the jacobian only once
- `-ksp_rmonitor` monitor for ksp solver
- `-snes_monitor` monitor for snes sover
- `-snes_converged_reason` convergence reason for snes solver
- `-ksp_converged_reason` convergence reason for ksp solver