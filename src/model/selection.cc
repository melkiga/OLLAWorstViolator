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

#include "selection.h"

Pattern::~Pattern() {
	delete [] coords;
}

Pattern* PatternFactory::createCross() {
	Pattern *cross = new Pattern();
	cross->spread = 2;
	cross->size = 5;
	TrainingCoord *coords = new TrainingCoord[5];
	coords[0].c = 0;
	coords[0].gamma = 0;
	coords[1].c = -1;
	coords[1].gamma = 0;
	coords[2].c = 0;
	coords[2].gamma = -1;
	coords[3].c = 1;
	coords[3].gamma = 0;
	coords[4].c = 0;
	coords[4].gamma = 1;
	cross->coords = coords;
	return cross;
}

GridGaussianModelSelector::~GridGaussianModelSelector() {
}


TestingResult GridGaussianModelSelector::validate(
  CrossValidationSolver& solver, fvalue c, fvalue gamma) {
    Timer timer;
	  CGaussKernel param(gamma);
    solver.setKernelParams(c, param);

    timer.start();
    TestingResult result = solver.doCrossValidation();
    timer.stop();

    //logger << format("outer fold %d CV: time=%.2f[s], accuracy=%.2f[%%], C=%.4g, G=%.4g\n")
    //  % solver.getOuterFold() % timer.getTimeElapsed() % (100.0 * result.accuracy) % c % gamma;

    return result;
}


ModelSelectionResults GridGaussianModelSelector::selectParameters(
  CrossValidationSolver &solver, SearchRange &range) {
    fvalue cRatio = LOG_STEP(range.cLow, range.cHigh, range.cResolution);
    fvalue gammaRatio = LOG_STEP(range.gammaLow, range.gammaHigh, range.gammaResolution);

    ModelSelectionResults results;
    results.bestResult.accuracy = 0.0;

    for (quantity cIter = 0; cIter < range.cResolution; cIter++) {
      fvalue c = range.cLow * pown(cRatio, cIter);
      for (quantity gammaIter = 0; gammaIter < range.gammaResolution; gammaIter++) {
        fvalue gamma = range.gammaLow * pown(gammaRatio, gammaIter);

        TestingResult result = validate(solver, c, gamma);

        if (result.accuracy > results.bestResult.accuracy) {
          results.bestResult = result;
          results.c = c;
          results.gamma = gamma;
        }
      }
    }

    return results;
}


TestingResult GridGaussianModelSelector::doNestedCrossValidation(
  CrossValidationSolver &solver, SearchRange &range) {
    Timer timer;

    TestingResult result;
	  CGaussKernel initial(range.gammaLow);
    solver.setKernelParams(range.cLow, initial);
    for (fold_id fold = 0; fold < solver.getOuterFoldsNumber(); fold++) {
      solver.resetOuterFold(fold);

      timer.restart();
      ModelSelectionResults params = selectParameters(solver, range);
      timer.stop();

      //logger << format("outer fold %d model selection: time=%.2f[s], accuracy=%.2f[%%], C=%.4g, G=%.4g\n")
      //  % fold % timer.getTimeElapsed() % (100.0 * params.bestResult.accuracy) % params.c % params.gamma;

	    CGaussKernel kernel(params.gamma);
      solver.setKernelParams(params.c, kernel);

      timer.restart();
      solver.trainOuter();
      timer.stop();

      StateHolder &stateHolder = solver.getStateHolder();
      //logger << format("outer fold %d final training: time=%.2f[s], sv=%d/%d\n")
      //  % fold % timer.getTimeElapsed() % stateHolder.getSvNumber() % solver.getOuterProblemSize();

      timer.restart();
      TestingResult current = solver.testOuter();
      timer.stop();

      //logger << format("outer fold %d final testing: time=%.2f[s], accuracy=%.2f[%%]\n")
      //  % fold % timer.getTimeElapsed() % (100.0 * current.accuracy);

      result.accuracy += current.accuracy / solver.getOuterFoldsNumber();
    }
    return result;
}


PatternGaussianModelSelector::PatternGaussianModelSelector(Pattern *pattern) :
  pattern(pattern) {
}


PatternGaussianModelSelector::~PatternGaussianModelSelector() {
  delete pattern;
}


void PatternGaussianModelSelector::registerResult(TestingResult result, offset c, offset gamma) {
  results[TrainingCoord(c, gamma)] = result;
}


quantity PatternGaussianModelSelector::evaluateDistance(offset c, offset gamma, SearchRange &range) {
  //	quantity dist = min(min(c, gamma), min(range.cResolution - c, range.gammaResolution - gamma) - 1);
  quantity dist = range.cResolution + range.gammaResolution;
  map<TrainingCoord, TestingResult>::iterator it;
  for (it = results.begin(); it != results.end(); it++) {
    TrainingCoord coord = it->first;
    quantity cdiff = (c > coord.c) ? (c - coord.c) : (coord.c - c);
    quantity sdiff = (gamma > coord.gamma) ? (gamma - coord.gamma) : (coord.gamma - gamma);
    dist = min(dist, cdiff + sdiff);
  }
  return dist;
}


TrainingCoord PatternGaussianModelSelector::findStartingPoint(SearchRange &range) {
  TrainingCoord startingPoint(INVALID_OFFSET, INVALID_OFFSET);

  offset cCenterOffset = (range.cResolution - 1) / 2;
  offset gammaCenterOffset = (range.gammaResolution - 1) / 2;

  quantity maxDist = 0;

  if (!results.empty()) {
    quantity rangeSpread = min(range.cResolution, range.gammaResolution);
    quantity scale = (quantity) max(exp2(floor(log2((rangeSpread - 1) / pattern->spread))), 1.0);
    quantity minDist = (quantity) ceil(sqrt(rangeSpread) / 2);
    do {
      for (offset c = cCenterOffset % scale; c < range.cResolution; c += scale) {
        for (offset s = gammaCenterOffset % scale; s < range.gammaResolution; s += scale) {
          quantity dist = evaluateDistance(c, s, range);

          if (dist > maxDist) {
            maxDist = dist;
            if (dist >= minDist) {
              startingPoint.c = c;
              startingPoint.gamma = s;
            }
          }
        }
      }
      scale /= 2;
    } while (scale > minDist);
  } else {
    startingPoint.c = cCenterOffset;
    startingPoint.gamma = gammaCenterOffset;
  }
  return startingPoint;
}


ModelSelectionResults PatternGaussianModelSelector::selectParameters(
  CrossValidationSolver &solver, SearchRange &range) {
    results.clear();

    ModelSelectionResults globalRes;
    globalRes.bestResult.accuracy = 0.0;

    TrainingCoord trnCoords = findStartingPoint(range);
    while (trnCoords.c != INVALID_OFFSET && trnCoords.gamma != INVALID_OFFSET) {
      quantity rangeSpread = min(range.cResolution, range.gammaResolution);
      quantity scale = (quantity) max(exp2(floor(log2((rangeSpread - 1) / pattern->spread))), 1.0);
      offset cOffset = trnCoords.c;
      offset gammaOffset = trnCoords.gamma;

      fvalue cRatio = LOG_STEP(range.cLow, range.cHigh, range.cResolution);
      fvalue gammaRatio = LOG_STEP(range.gammaLow, range.gammaHigh, range.gammaResolution);

      while (scale > 0) {
        TrainingCoord bestPosition;
        fvalue bestAccuracy = -MAX_FVALUE;

        for (quantity i = 0; i < pattern->size; i++) {
          TrainingCoord shift = pattern->coords[i];
          TrainingCoord current(cOffset + scale * shift.c, gammaOffset + scale * shift.gamma);

          if (current.c >= 0 && current.c < range.cResolution
            && current.gamma >= 0 && current.gamma < range.gammaResolution) {
              TestingResult result;

              if (results.find(current) == results.end()) {
                fvalue c = range.cLow * pown(cRatio, current.c);
                fvalue gamma = range.gammaLow * pown(gammaRatio, current.gamma);

                result = this->validate(solver, c, gamma);
                registerResult(result, current.c, current.gamma);
              } else {
                result = results[current];
              }

              if (result.accuracy > bestAccuracy) {
                bestPosition = current;
                bestAccuracy = result.accuracy;
              }
          }
        }

        if (bestAccuracy > globalRes.bestResult.accuracy) {
          globalRes.bestResult.accuracy = bestAccuracy;
          globalRes.c = range.cLow * pown(cRatio, bestPosition.c);
          globalRes.gamma = range.gammaLow * pown(gammaRatio, bestPosition.gamma);
        }

        if (cOffset == bestPosition.c && gammaOffset == bestPosition.gamma) {
          scale /= 2;
        } else {
          cOffset = bestPosition.c;
          gammaOffset = bestPosition.gamma;
        }
      }
      trnCoords = findStartingPoint(range);
    }
    return globalRes;
}
