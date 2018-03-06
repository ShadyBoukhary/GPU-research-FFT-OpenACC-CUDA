# GPU-research-FFT-OpenACC-CUDA 

A performance analysis of the implementation of Cooley-Turkey algorithm in order to solve radix-2 Fast Fourier Transform problem using sequential code and parallel code in OpenACC and CUDA. The performance between the 3 methods is analyzed, especially between openACC and CUDA to determine the efficiency of openACC code compared to low level optimized CUDA code.

Project starts off with smaller tests between the different methods.

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

### Installing (Adapted from http://www.pgroup.com/doc/pgiinstall174.pdf)

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

Make PGI products accessible.

```
$ export PGI=/opt/pgi;
$ export PATH=/opt/pgi/linux86-64/17.4/bin:$PATH;
$ export MANPATH=$MANPATH:/opt/pgi/linux86-64/17.4/man;
$ export LM_LICENSE_FILE=$LM_LICENSE_FILE:/opt/pgi/license.dat; 
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

