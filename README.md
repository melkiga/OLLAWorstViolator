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

- **CMake** [install](https://cmake.org/download/)
- **Boost** [install](https://www.boost.org/users/download/)
- **GSL** [install](https://www.gnu.org/software/gsl/)
- **VSCode** [install](https://code.visualstudio.com/docs/cpp/config-linux)
  - Extension C/C++ - `ms-vscode.cpptools`
  - Extension CMake - `ms-vscode.cmake-tools`

## Authors

- **Gabriella Melki** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/melkiga/OLLAWorstViolator/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Hat tip to Robert Strack.
- All the love for Vojo.
- Inspiration from Andrew Ritz.
