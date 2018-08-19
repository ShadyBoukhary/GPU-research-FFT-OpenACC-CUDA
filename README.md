# Evaluating the Benefits of Many-Core Programming Models: A Case Study

A performance analysis of the implementation of Cooley-Turkey algorithm in order to solve radix-2 Fast Fourier Transform problem using sequential code and parallel code in OpenACC and CUDA. The performance between the 3 methods is analyzed, especially between openACC and CUDA to determine the efficiency of openACC code compared to low level optimized CUDA code.

Project starts off with smaller tests between the different methods.

## Results

* [Radix-2 FFT Sequential VS OpenACC VS CUDA Global Memory VS CUDA Shared Memory](Week%205/FFT%20Seq%20vs%20OpenACC%20vs%20GM%20vs%20SM.pdf)
* [Matrix Multiplication Sequential VS OpenACC VS CUDA Global Memory VS CUDA Shared Memory](Week%205/Matrix%20Mult.%20Seq%20vs%20OpenACC%20vs%20Cuda.pdf)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites


```
Latest Nvidia drivers
CUDA 9.0 Toolkit
GCC and G++ compilers
OpenACC PGI Compiler
PGI Account
```

### Installing [Adapted from here](http://www.pgroup.com/doc/pgiinstall174.pdf)

#### Linux

Linux Standard Base, lsb, must be present for the license utilities to work properly.
Enter the command lsb_release to see if you have lsb, and, if so, which version.
PGI installations require version 3.0 or newer. To install lsb, try running one of the
following command

```
$ sudo apt–get install lsb
```

Create an account at www.pgroup.com.
Go to www.pgroup.com/register to create an account.

When ready, you can download the proper package(s) at www.pgroup.com/
support/downloads.php. The file sizes can be large. Will be used later.

Install Nvidia CUDA 9.0 Toolkit from https://developer.nvidia.com/cuda-downloads

Unpack the PGI software.
In the instructions that follow, replace <tarfile> with the name of the file that you
downloaded.
Use the following command sequence to unpack the tar file before installation.

```
$ tar xpfz <tarfile>.tar.gz
```

Run the installation script and follow on-screen instructions.

```
$ sudo ./install
```

Make PGI products accessible. In the terminal:
***NOTE: the directory with the version number 17.10 may vary according to your installation. Verify that it is correct or change accordingly***

```
export PGI=/opt/pgi;
export PATH=/opt/pgi/linux86-64/17.10/bin:$PATH;
export MANPATH=$MANPATH:/opt/pgi/linux86-64/17.10/man;
export LM_LICENSE_FILE=$LM_LICENSE_FILE:/opt/pgi/license.dat; 
```

That should be it for the community version.

#### WIndows

The Microsoft Windows Software Development Kit (SDK) is a prerequisite for all
Windows installs. Download the Windows SDK at http://www.pgroup.com/microsoftsdk.

The PGI Community Edition on Windows requires that Microsoft Visual Studio 2015
with Visual C++ be installed first. 

Install CUDA Toolkit and drivers just like the linux instructions.

Download the software from www.pgroup.com or another electronic distribution site.
Select the install package appropriate for your OS.

Run the installation executables as Administrator.

That should be it for the community version.



## Authors

* **Shady Boukhary** 


## References
* [OpenACC Programming Guide](http://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0.pdf)
* [Running OpenACC Programs on Nvidia and AMD GPUs](https://www.pgroup.com/lit/presentations/ieee_webinar_dec2013_slides.pdf)
* [Getting Started with OpenACC](https://www.pgroup.com/doc/openacc17_gs.pdf)
* [OpenACC Compiler Directives](https://www.pgroup.com/lit/brochures/openacc_sc14.pdf)
* [OpenACC Full Installation Guide](http://www.pgroup.com/doc/pgiinstall174.pdf)
