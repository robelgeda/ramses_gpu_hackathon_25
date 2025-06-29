# Sprint 

## Goals 
Here is a list of tasks we will discuss tomorrow.
The idea is to divide up the group in teams so we can prepare for day 1 next week.
- Write a 3x3x3 kernel version from the existing 4x4x4 octs arrangement in the gpu folder
- Write the f90 wrapper that deals with the ghost octs and send the data to the gpu (assuming the 3x3x3 kernel is ready).
- Execute the existing gpu code on stellar gpu
- Profile the existing gpu code sing Nsight Systems and Compute
We can split the team in 3 or 4 groups dealing with these aspects in parallel.
Feel free to comment on slack.
Romain



## Installation
You have to use the develop branch.

To compile: `make NDIM=3 COMPILER=NVHPC PATCH=../gpu`


## Running 
> **_NOTE:_**
For PU-NVIDIA Open Hackathon: We will have reserved GPU nodes on Della and Stellar beginning June 4 2025. Use the Slurm directive below to access these nodes: `#SBATCH --reservation=openhack`

Some useful information about using PU GPU can be found:

- https://github.com/PrincetonUniversity/gpu_programming_intro/tree/master/03_your_first_gpu_job
- https://researchcomputing.princeton.edu/support/knowledge-base/slurm#gpus

### To run on Della:
- You must log on `della-gpu`
- `module load nvhpc-hpcx-cuda12/25.5`
- cd bin
- `make NDIM=3 COMPILER=NVHPC PATCH=../gpu`
- cd ..
- `salloc --nodes=1 --ntasks=1 --time=01:00 --gres=gpu:1`
- Once on the compute node: `bin/ramses3d namelist/sedov3d.nml`


If you want to use openmp, do this:
salloc --nodes=1 --ntasks=48 --time=00:01:00 --gres=gpu:1
export OMP_NUM_THREADS=48
bin/ramses3d namelist/sedov3d.nml

### Run mni-ramses on stellar using gpu.
- You need to log on `stellar-amd`
- `module load nvhpc-hpcx-cuda12/25.5`
- `make NDIM=3 COMPILER=NVHPC PATCH=../gpu`
- `salloc --nodes=1 --ntasks=1 --time=00:01:00 --gres=gpu:1`
- `bin/ramses3d namelist/sedov3d.nml`

You can however compile the code with -gpu=cc70 and run directly on the V100 on stellar-vis1


## Notes 

### Setting CUDA device

To set the active CUDA device in Fortran, you can use the `cuda_set_device` function. This function is part of the CUDA Fortran runtime library, and it's used to select the specific GPU you want to use for calculations. Here's how you would use it: [1]
```
program main
    use cudaf
    implicit none

    integer :: device_id

    ! Set the active CUDA device
    device_id = 0  ! Example: Set device ID to 0 (the default)

    call cuda_set_device(device_id)

    ! ... rest of your code ...

end program main
```

**Explanation:**

- `use cudaf`: This line imports the CUDA Fortran module, which contains the interface for the `cuda_set_device` function.
- `implicit none`: This line ensures that all variables are explicitly declared.
- `integer :: device_id`: This declares an integer variable to hold the device ID.
- `device_id = 0`: This sets the device ID to 0. CUDA devices are numbered starting from 0. If you have multiple GPUs, you can select a different ID.
- `call cuda_set_device(device_id)`: This line calls the cuda_set_device function, passing the desired device ID as an argument. This sets the specified GPU as the active device for subsequent CUDA calls.
- `... rest of your code ...`: After setting the device, you can proceed with your CUDA Fortran code, which will now operate on the selected device. [2, 3, 4, 5]

AI responses may include mistakes.
[1] https://discuss.pytorch.org/t/how-to-change-the-default-device-of-gpu-device-ids-0/1041

[2] https://docs.nvidia.com/hpc-sdk/pgi-compilers/17.9/x86/cuda-fortran-prog-guide/index.htm

[3] https://faculty.washington.edu/rjl/classes/am583s2013/notes/fortran.html

[4] https://ppeetteerrsx.com/post/cuda/stylegan_cuda_kernels/

[5] https://wvuhpc.github.io/Modern-Fortran/31-CUDA-Fortran/index.html

With MPI
```
program cuda_mpi_example
    use mpi
    use cudafor
    implicit none
    integer :: ierr, rank, size
    integer :: device_id, num_devices
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    ! Get the number of available CUDA devices
    call cudaGetDeviceCount(num_devices)
    ! Assign a device to each MPI process (wrap around if more processes than devices)
    device_id = mod(rank, num_devices)
    ! Set the CUDA device for this MPI process
    call cudaSetDevice(device_id)
    print *, 'MPI rank', rank, 'using CUDA device', device_id
    ! ... your CUDA Fortran code here ...
    call MPI_Finalize(ierr)
end program cuda_mpi_example
```


### Nsight Systems Reports

In case it's useful to anyone, here are the commands I used to generate the Nsight Systems reports I showed in the zoom meeting earlier.

After compiling ramses3d with module load nvhpc-hpcx-cuda12/25.5 and make NDIM=3 COMPILER=NVHPC PATCH=../gpu :
1. `salloc --nodes=1 --ntasks=1 --time=00:03:00 --gres=gpu:1`
2. `module load nsight-systems/2025.3.1`
3. `nsys profile --trace=cuda,nvtx,osrt --stats=true --output=sedov3d bin/ramses3d namelist/sedov3d.nml`

Then you can open the created sedov3d.nsys-rep file in the user interface to see the report.

For the Nsight Compute reports replace the 2nd and 3rd steps by

2. `module load nvhpc-hpcx-cuda12/25.5`
3. `ncu --set=full --export=sedov3d --target-processes all bin/ramses3d namelist/sedov3d.nml`
Here the file to open in the user interface is `sedov3d.ncu-rep`

### Bob Caddy's Aliases

```shell
alias nsysRamses="nsys profile --trace=cuda,nvtx,openmp,mpi ./ramses3d ../namelist/sedov3d.nml"
alias ncuRamses="ncu -o profile --set full ./ramses3d ../namelist/sedov3d.nml"
alias memcheckRamses="compute-sanitizer  --tool memcheck  --print-level info --report-api-errors all ./ramses3d ../tests/namelist/sedov3d.nml"
alias racecheckRamses="compute-sanitizer --tool racecheck --print-level info ./ramses3d ../tests/namelist/sedov3d.nml"
alias synccheckRamses="compute-sanitizer --tool synccheck --print-level info ./ramses3d ../tests/namelist/sedov3d.nml"
alias initcheckRamses="compute-sanitizer --tool initcheck --print-level info --track-unused-memory ./ramses3d ../tests/namelist/sedov3d.nml"
```
 
 ### Links 
 - If you need CUDA Fortran intro, my work-mates Greg and Mass have a book out, newly refreshed.
https://shop.elsevier.com/books/cuda-fortran-for-scientists-and-engineers/ruetsch/978-0-443-21977-1