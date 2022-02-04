#Download petsc
git clone -b release https://gitlab.com/petsc/petsc.git petsc

# Install
$ cd petsc

$ ./configure --with-debugging=0 --with-debugging=0 --download-p4est=1 --with-zlib=1 --with-fortran-bindings=0 --download-parmetis=1 --download-metis=1 -download-hdf5=1 --download-hypre=1 --download-scalapack --download-superlu --download-superlu_dist --download-mumps --with-petsc4py

$ make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-linux-c-opt all


