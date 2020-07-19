# Project Title

This repository contains the implementation of OLLAWV, an online Support Vector Machine solver. Copyright (C) 2018-2020 Gabriella Melki (gabriellamelki@gmail.com), Vojislav Kecman (kecmanv@vcu.edu), All Rights Reserved. If you find this code useful, please cite the folowing:

```bibtex
@article{melki2018ollawv,
  title={OLLAWV: online learning algorithm using worst-violators},
  author={Melki, Gabriella and Kecman, Vojislav and Ventura, Sebastian and Cano, Alberto},
  journal={Applied Soft Computing},
  volume={66},
  pages={384--393},
  year={2018},
  publisher={Elsevier}
}
```

## Introduction

To address the limitations presented by current popular SVM solvers, we developed novel OnLine Learning Algorithm using Worst-Violators (OLLAWV). The key contributions of OLLAWV include:

- A unique iterative procedure for solving the L1-SVM problem, as well as a novel method for identifying support vectors, or worst-violators. Rather than randomly iterating over the data samples, OLLAWV aims to reduce training time by selecting and updating the samples that are most incorrectly classified with respect to the current decision hyper-plane.

- A novel stopping criteria by utilizing the worst-violator identification method. This aims to eliminate the added parameterization that is included with most online methods, where the number of iterations of the algorithm needs to be set in advance. Once there are no incorrectly classified samples left, the algorithm terminates.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. The project uses VSCode as it's IDE and CMake for building and running.

### Installs

- **CMake** [install](https://cmake.org/download/)
- **VSCode** [install](https://code.visualstudio.com/docs/cpp/config-linux)
  - Extension C/C++ - `ms-vscode.cpptools`
  - Extension CMake - `ms-vscode.cmake-tools`
- **Boost** [install](https://www.boost.org/users/download/)
  - **Windows** [getting-started](https://www.boost.org/doc/libs/1_73_0/more/getting_started/windows.html#boost-root-directory) | [cmake-test](https://www.boost.org/doc/libs/1_73_0/libs/test/doc/html/boost_test/adv_scenarios/build_utf.html)
    1. Download and extract Boost
    2. Add the following path to your environment variables `C:\Boost`
    3. Navigate to the newly extracted Boost folder within your terminal/cmd
    4. Run the following:

        ```cmd
        bootstrap.bat
        .\b2 address-model=32 architecture=x86 --build-type=complete --with-program_options --with-test toolset=msvc link=shared install 
        ```

  - **Linux**

      ```bash
      libboost-dev/focal,now 1.71.0.0ubuntu2 amd64 [installed,automatic]
      libboost-dev/focal 1.71.0.0ubuntu2 i386
      ```

- **GSL** [windows](https://solarianprogrammer.com/2020/01/26/getting-started-gsl-gnu-scientific-library-windows-macos-linux/) | [linux](https://www.gnu.org/software/gsl/)

  - **Windows**:
    - `git clone https://github.com/microsoft/vcpkg.git`
    - `cd vcpkg`
    - `.\bootstrap-vcpkg.bat`
    - `.\vcpkg integrate install`
    - `.\vcpkg install gsl gsl:x64-windows`
    - Add environment variable `GSL_ROOT` with value: `C:path\to\vcpkg\packages\gsl_x64-windows`
    - For more information on *vcpkg* visit: https://vcpkg.readthedocs.io/en/latest/

  - **Linux**:

    ```bash
    libgsl-dev/focal,now 2.5+dfsg-6build1 amd64 [installed]
    libgsl-dev/focal 2.5+dfsg-6build1 i386
    ```

### Configure

[Configure](https://code.visualstudio.com/docs/cpp/c-cpp-properties-schema-reference) `VSCode` with your preferred C++ IDE settings.

## Setup

1. Clone this repository.
2. Build `Ctrl+Shift+B` or use the `VSCode CMake` extension.
   1. Be sure to select your config. These are already set up in the Launch.json file or CMakeLists.txt
3. Debug, Run, or Test
   1. Debug: `F5`
   2. Run: enter the following in the terminal

```bash
bin/Release/osvm -i 5 -o 5 /path/to/data # Run
test/bin/Debug/osvm_unit_tests -i 5 -i 5 -d true -t /path/to/output/json /path/to/data # Test
```

## Project Details

```bash
.
├── bin                   # Debug/osvm and Release/osvm executables
├── build                 # cmake dir
├── src                   # source code files
│   ├── data
│   ├── feature
│   ├── logging
│   ├── math
│   ├── model
│   ├── svm
│   ├── time
│   ├── configuration.cc
│   ├── configuration.h
│   ├── launcher.cc
│   ├── launcher.h
│   ├── osvm.cc           # main file
│   └── osvm.h
├── test                  # boost unit tests
│   ├── bin               # Debug/osvm and Release/osvm test executables
│   ├── examples          # json files with true example outputs
│   ├── CMakeLists.txt    # test cmake config
│   ├── osvm_test.cc      # main test file
│   └── osvm_test.h
├── .vscode               # vscode configurations
│   ├── c_cpp_properties.json # c++ configurations (edit to your liking)
│   ├── launch.json       # launch/attach configurations for debugging
│   ├── settings.json     # general vscode settings 
│   └── tasks.json        # build instructions for each release (debug,release,test)
├── CMakeLists.txt        # cmake config for main project
├── .gitattributes
├── .gitignore
├── IssueTemplate.md
└── README.md
```

## Authors

- **Gabriella Melki**

See also the list of [contributors](https://github.com/melkiga/OLLAWorstViolator/contributors) who participated in this project.

## Acknowledgments

- Hat tip to Robert Strack.
- All the love for Vojo.
- Inspiration from Andrew Ritz.
