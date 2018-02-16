This file is part of osvm, an online Support Vector Machine solver.
Copyright (C) 2017-2018 Gabriella Melki (melkiga@vcu.edu), Vojislav Kecman (kecmanv@vcu.edu), All Rights Reserved.
**************************************************************************
* This repo contains VS 2015 Project for OnLine Learning Algorithm using Worst Violators (OLLAWV). The VS solution is dependent on 
two external libraries: boost and gsl. The input data format that it takes is LIBSVM format. 

* Setting up your build environment:
  * Download & Install Boost library (x64)
  * Download & Install GSL library (x64)
  * Define the following environment variables to point to the Boost and GSL libraries mentioned above.
    * OLLAWV_GSL = <drive:\path\to\gsl>
    * OLLAWV_BOOST = <drive:\path\to\boost>
    * Restart VS if it is currently running.
* The VS project configuration will not need to be modified, but notable settings are:
    * C/C++ - Additional Include Directories - 
      - ($OLLAWV_GSL)\x64\include\
      - ($OLLAWV_BOOST)\x64\
    * Linker - Additional Library Directories -
      - ($OLLAWV_GSL)\x64\lib
      - ($OLLAWV_BOOST)\x64\stage\lib
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
  


