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

#ifndef LAUCHER_H_
#define LAUCHER_H_

#include "configuration.h"

class ApplicationLauncher {

	Configuration &conf;
	Classifier* run();

protected:
	template<typename Strategy>
	CrossValidationSolver<Strategy>* createCrossValidator();

	template<typename Strategy>
	AbstractSolver<Strategy>* createSolver();

	template<typename Strategy>
	GridGaussianModelSelector<Strategy>* createModelSelector();

	template<typename Strategy>
	Classifier* performTraining();

	template<typename Strategy>
	Classifier* performCrossValidation();

	template<typename Strategy>
	Classifier* performModelSelection();

	template<typename Strategy>
	Classifier* performNestedCrossValidation();

public:
	ApplicationLauncher(Configuration &conf) : conf(conf) {
	}

	void launch();

};

template<typename Strategy>
CrossValidationSolver<Strategy>* ApplicationLauncher::createCrossValidator() {
	ifstream input(conf.dataFile.c_str());
	BaseSolverFactory<Strategy> reader(
			input, conf.trainingParams, conf.stopCriterion);

	Timer timer(true);
	CrossValidationSolver<Strategy> *solver
			= (CrossValidationSolver<Strategy>*) reader.getCrossValidationSolver(
					conf.validation.innerFolds, conf.validation.outerFolds);
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();
	input.close();
	return solver;
}

template<typename Strategy>
AbstractSolver<Strategy>* ApplicationLauncher::createSolver() {
	ifstream input(conf.dataFile.c_str());
	BaseSolverFactory<Strategy> reader(input, conf.trainingParams, conf.stopCriterion);

	Timer timer(true);
	AbstractSolver<Strategy> *solver = reader.getSolver();
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();
	input.close();
	return solver;
}

template<typename Strategy>
GridGaussianModelSelector<Strategy>* ApplicationLauncher::createModelSelector() {
	GridGaussianModelSelector<Strategy> *selector;
	if (conf.validation.modelSelection == GRID) {
		selector = new GridGaussianModelSelector<Strategy>();
	} else {
		PatternFactory factory;
		selector = new PatternGaussianModelSelector<Strategy>(factory.createCross());
	}
	return selector;
}

template<typename Strategy>
Classifier* ApplicationLauncher::performModelSelection() {
	CrossValidationSolver<Strategy> *solver
			= createCrossValidator<Strategy>();

	Timer timer(true);
	GridGaussianModelSelector<Strategy> *selector
			= createModelSelector<Strategy>();
	ModelSelectionResults params = selector->selectParameters(*solver, conf.searchRange);
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%], C=%.4g, G=%.4g\n")
			% timer.getTimeElapsed() % (100.0 * params.bestResult.accuracy)
			% params.c % params.gamma;
	
	delete selector;
	return solver->getClassifier();
}

template<typename Strategy>
Classifier* ApplicationLauncher::performNestedCrossValidation() {
	CrossValidationSolver<Strategy> *solver
			= createCrossValidator<Strategy>();

	Timer timer(true);
	GridGaussianModelSelector<Strategy> *selector
			= createModelSelector<Strategy>();
	TestingResult res = selector->doNestedCrossValidation(*solver, conf.searchRange);
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * res.accuracy);

	delete selector;
	return solver->getClassifier();
}

template<typename Strategy>
Classifier* ApplicationLauncher::performCrossValidation() {
	CrossValidationSolver<Strategy> *solver
			= createCrossValidator<Strategy>();

	Timer timer(true);
	CGaussKernel param(conf.searchRange.gammaLow);
	solver->setKernelParams(conf.searchRange.cLow, param);
	TestingResult result = solver->doCrossValidation();
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * result.accuracy);

	return solver->getClassifier();
}

template<typename Strategy>
Classifier* ApplicationLauncher::performTraining() {
	AbstractSolver<Strategy> *solver = createSolver<Strategy>();

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

	delete solver;
	return classifier;
}

#endif
