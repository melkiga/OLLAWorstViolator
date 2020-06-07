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

void ApplicationLauncher::run() {
  Classifier* classifier;
  if (conf.validation.outerFolds > 1) {
    classifier = performNestedCrossValidation();
  }
  else {
    if (conf.validation.innerFolds > 1) {
      if (conf.searchRange.cResolution > 1 || conf.searchRange.gammaResolution > 1) {
        classifier = performModelSelection();
      }
      else {
        classifier = performCrossValidation();
      }
    }
    else {
      classifier = performTraining();
    }
  }

  // TODO: add function for printing classifier to file
}

// TODO: fill out saving classifier here. 
// this should be where we save the classifier to a file, but
// because pairwise classifier is a template, i can't do it here.
void ApplicationLauncher::saveClassifier(Classifier* classifier){
  int numberSupportVectors = classifier->getSvNumber();
}

CrossValidationSolver* ApplicationLauncher::createCrossValidator() {
	ifstream input(conf.dataFile.c_str());
	BaseSolverFactory reader(input, conf.trainingParams, conf.stopCriterion);

	Timer timer(true);
	CrossValidationSolver *solver
			= (CrossValidationSolver*) reader.getCrossValidationSolver(
					conf.validation.innerFolds, conf.validation.outerFolds);
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();
	input.close();
	return solver;
}

AbstractSolver* ApplicationLauncher::createSolver() {
	ifstream input(conf.dataFile.c_str());
	BaseSolverFactory reader(input, conf.trainingParams, conf.stopCriterion);

	Timer timer(true);
	AbstractSolver *solver = reader.getSolver();
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();
	input.close();
	return solver;
}

GridGaussianModelSelector* ApplicationLauncher::createModelSelector() {
	GridGaussianModelSelector *selector;
	if (conf.validation.modelSelection == GRID) {
		selector = new GridGaussianModelSelector();
	} else {
		PatternFactory factory;
		selector = new PatternGaussianModelSelector(factory.createCross());
	}
	return selector;
}

Classifier* ApplicationLauncher::performModelSelection() {
	CrossValidationSolver *solver
			= createCrossValidator();

	Timer timer(true);
	GridGaussianModelSelector *selector
			= createModelSelector();
	ModelSelectionResults params = selector->selectParameters(*solver, conf.searchRange);
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%], C=%.4g, G=%.4g\n")
			% timer.getTimeElapsed() % (100.0 * params.bestResult.accuracy)
			% params.c % params.gamma;
	
	delete selector;
	return solver->getClassifier();
}

Classifier* ApplicationLauncher::performNestedCrossValidation() {
	CrossValidationSolver *solver = createCrossValidator();

	Timer timer(true);
	GridGaussianModelSelector *selector = createModelSelector();
	TestingResult res = selector->doNestedCrossValidation(*solver, conf.searchRange);
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * res.accuracy);

	delete selector;
	return solver->getClassifier();
}

Classifier* ApplicationLauncher::performCrossValidation() {
	CrossValidationSolver *solver = createCrossValidator();

	Timer timer(true);
	CGaussKernel param(conf.searchRange.gammaLow);
	solver->setKernelParams(conf.searchRange.cLow, param);
	TestingResult result = solver->doCrossValidation();
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * result.accuracy);

	return solver->getClassifier();
}

Classifier* ApplicationLauncher::performTraining() {
	AbstractSolver *solver = createSolver();

	Timer timer(true);
	CGaussKernel param(conf.searchRange.gammaLow);
	solver->setKernelParams(conf.searchRange.cLow, param);
	solver->train();
	Classifier* classifier = solver->getClassifier();
	timer.stop();

	sfmatrix *samples = solver->getSamples();
	label_id *labels = solver->getLabels();
	quantity correct = 0;
	quantity total = (quantity) samples->height;
	for (sample_id v = 0; v < total; v++) {
		label_id lbl = classifier->classify(v);
		if (lbl == labels[v]) {
			correct++;
		}
	}

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%], sv=%d\n")
			% timer.getTimeElapsed() % (100.0 * correct / total)
			% classifier->getSvNumber();

	return classifier;
}