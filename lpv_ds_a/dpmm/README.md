# Dirichlet Process Mixture Model (Parallel Implementation)

This module consists of the parallel implementation of Dirichlet process Mixture model(DPMM) that has been optimized for real-time learning performance. Given a set of demonstration trajectories, this module performs unsupervised learning of Gaussian mixture models that best describe the structure of provided data. In addition to the general clustering purposes, this module serves as an intermediate step in the pipeline of learning a Dynamical system-based motion policies from data, and the learned model will proceed to be optimized in the linear parameter varying(LPV) learning of a dynamical system.

--- 

### Update

3/29: log in the value of joint log likelihood in each iteration; can potentially increase computational load; visualize for convergence analysis


As opposed to using CMake in previous iteration, Linux implementation will solely use terminal commands to compile, execute and debug in Vsode IDE due to the module's low complexity.


---

### Dependencies
- **[Required]** [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page): Eigen library provides fast and efficient implementations of matrix operations and other linear algebra tools. The latest stable release, Eigen 3.4 is required.
- **[Required]** [Boost](https://www.boost.org/): Boost provides portable C++ source libraries that works efficiently with standard C++ libraries. The latest release Boost 1.81 is recommended.
- **[Required]** [OpenMP](https://www.openmp.org/): OpenMP allows the parallel computation in C++ platform.

---

### Installation
To install Eigen and Boost libraries, please download both packages directly from their websites and extract the downloaded zip files under the system include path: `/usr/include/`

OpenMP usually comes with the GCC compiler starting from its version 4.2.0. Check the GCC version of the computer:

```gcc --version```

If the system doesn’t have the GCC compiler, please install using the following command


```sudo apt install gcc```

---

### Compilation

Eigen library is made of standalone header files, and hence no additional linking and compilation is needed other than including its path. On the other hand, while most of the Boost functionalities are entirely defined in header files, certain packages require build. We need to first build a binary library from boost before compiling the main source code:

First located the boost library in system path:

```cd /usr/include/boost_1.81_0```

Run the built-in build tools in boost:

```./bootstrap.sh  --with-libraries=program_options```

and then,

```sudo ./b2 install ```

<!-- The built shared library is a dynamic library. Hence, we need to add the environmental variables that allow the source code to locate the library at run time:

```LD_LIBRARY_PATH=/usr/include/boost_1_81_0/stage/lib```

```export $LD_LIBRARY_PATH``` -->

Now we can go back to the root directory of dpmm and compile the source code using the following command,

```cd dpmm```

```g++ -O1 -fopenmp -Iinclude -I/usr/include/eigen-3.4.0 -I/usr/include/boost_1_81_0 src/niwDir.cpp src/niw.cpp src/normal.cpp src/normalDir.cpp src/dpmm.cpp src/dpmmDir.cpp  src/main.cpp -o main -lboost_program_options```


And execute the code 

```python main.py  [-d DATA] [-t ITERATION] [-a ALPHA] [--init INIT]```

<!-- 
GCC can search for package under system directory, but both packages have unconventional names with version information, we need to specify the include path for GCC to search using the -I flag 

Eigen library is completely header-based; hence no separate compilation is needed and can be directly referenced and used once the include path is specified.

On the other hand, boost library; while most of its functionalities are defined in header files, some packages do require separate compilation and linking; e.g., boost::program_options.

Use the built-in build system from the boost:
./bootstrap.sh --help
Also, consider using the --show-libraries and --with-libraries=library-name-list options to limit the long wait you'll experience if you build everything. 
./b2 install 

The binary library, if not specified, by default will be installed under the directory usr/include/boost_1_81_0/stage/lib. Make sure then use the -L flag to specifiy the library path and use the -l flag to search for the specific library in the path -->


