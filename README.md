This file is part of osvm, an online Support Vector Machine solver.
Copyright (C) 2017 Gabriella Melki (melkiga@vcu.edu), Vojislav Kecman (kecmanv@vcu.edu)
**************************************************************************
* This repo contains VS 2015 Project for OnLine Learning Algorithm using Worst Violators (OLLAWV). The VS solution is dependent on 
two external libraries: boost and gsl. The input data format that it takes is LIBSVM format. 

* Setting up VS2015 project configuration:
  * Download & Install Boost library (x64)
  * Download & Install GSL library (x64)
  * In VS project configuration:
    * C/C++ - Additional Include Directories - 
      - ..\path\to\boost\x64\
      - ..\path\to\gsl\x64\include\
    * Linker - Additional Library Directories -
      - ..\path\to\boost\x64\stage\lib
      - ..\path\to\gsl\x64\lib
    * Linker - Input - 
      - cblas.lib
      - gsl.lib
      - legacy_stdio_definitions.lib

* For help message (and run options): 
  * Launch Command Prompt
  * Navigate to project folder - x64 - Debug/Release
  * Type OLLAWV (this will dispaly help message with run options)

* Example run from command line:
  * OLLAWV -u pairwise -i 5 -o 5 "C:\data\smalldata\iris"

* Entry point of program: osvm.cc

* The core OLLAWV Algorithm is here: solver.h::trainForCache.
It is dependent on these files mainly: cache.h, kernel.h
  


