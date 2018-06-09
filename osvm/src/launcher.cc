/**************************************************************************
 * This file is part of gsvm, a Support Vector Machine solver.
 * Copyright (C) 2012 Robert Strack (strackr@vcu.edu), Vojislav Kecman
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************/

#include "launcher.h"

Classifier* ApplicationLauncher::run() {
  Classifier* classifier;
  if (conf.validation.outerFolds > 1) {
    classifier = performNestedCrossValidation<SolverStrategy >();
  }
  else {
    if (conf.validation.innerFolds > 1) {
      if (conf.searchRange.cResolution > 1 || conf.searchRange.gammaResolution > 1) {
        classifier = performModelSelection<SolverStrategy >();
      }
      else {
        classifier = performCrossValidation<SolverStrategy >();
      }
    }
    else {
      classifier = performTraining<SolverStrategy >();
    }
  }
  return classifier;
}

void ApplicationLauncher::launch() {
  run();
}
