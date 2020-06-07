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

#ifndef SELECTION_H_
#define SELECTION_H_

#include "../time/timer.h"
#include "../logging/log.h"
#include "../svm/validation.h"
#include "../svm/kernel.h"

#include <cmath>

#define LOG_STEP(f, t, s) ((t > f && s > 1) ? (exp(log((t) / (f)) / ((s) - 1))) : 1.0)

struct SearchRange {

  fvalue cLow;
  fvalue cHigh;
  quantity cResolution;

  fvalue gammaLow;
  fvalue gammaHigh;
  quantity gammaResolution;

};


struct ModelSelectionResults {

  fvalue c;
  fvalue gamma;

  TestingResult bestResult;

};


class GridGaussianModelSelector {

protected:
  TestingResult validate(CrossValidationSolver& solver,
    fvalue c, fvalue gamma);

public:
  virtual ~GridGaussianModelSelector();

  virtual ModelSelectionResults selectParameters(
    CrossValidationSolver &solver, SearchRange &range);
  TestingResult doNestedCrossValidation(CrossValidationSolver &solver,
    SearchRange &range);

};


typedef unsigned int offset;

#define INVALID_OFFSET ((offset) -1)

struct TrainingCoord {

  offset c;
  offset gamma;

  TrainingCoord(offset c = 0, offset gamma = 0) :
    c(c), gamma(gamma) {
  }

  bool operator<(const TrainingCoord &crd) const {
    return c > crd.c || (c == crd.c && gamma > crd.gamma);
  }

};


struct Pattern {

  TrainingCoord *coords;

  quantity size;
  quantity spread;

  ~Pattern();

};


class PatternFactory {

public:
  Pattern* createCross();

};



class PatternGaussianModelSelector: public GridGaussianModelSelector {

  Pattern *pattern;

  map<TrainingCoord, TestingResult> results;

protected:
  void registerResult(TestingResult result, offset c, offset gamma);

  TrainingCoord findStartingPoint(SearchRange &range);
  quantity evaluateDistance(offset c, offset gamma, SearchRange &range);

  void printTrainingMatrix(SearchRange &range);

public:
  PatternGaussianModelSelector(Pattern *pattern);
  virtual ~PatternGaussianModelSelector();

  virtual ModelSelectionResults selectParameters(
    CrossValidationSolver &solver,
    SearchRange &range);

};


#endif
