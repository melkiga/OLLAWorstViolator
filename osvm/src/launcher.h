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

	void selectMatrixTypeAndRun();

	template<typename Matrix>
	void selectViolationCriterionAndRun();

	template<typename Matrix, ViolationCriterion Violation>
	void selectGeneratorTypeAndRun();

	template<typename Matrix, ViolationCriterion Violation, GeneratorType Generator>
	void run();

protected:
	template<typename Matrix, typename Strategy>
	CrossValidationSolver<CGaussKernel, Matrix, Strategy>* createCrossValidator();

	template<typename Matrix, typename Strategy>
	AbstractSolver<CGaussKernel, Matrix, Strategy>* createSolver();

	template<typename Matrix, typename Strategy>
	GridGaussianModelSelector<Matrix, Strategy>* createModelSelector();

	template<typename Matrix, typename Strategy>
	void performTraining();

	template<typename Matrix, typename Strategy>
	void performCrossValidation();

	template<typename Matrix, typename Strategy>
	void performModelSelection();

	template<typename Matrix, typename Strategy>
	void performNestedCrossValidation();

public:
	ApplicationLauncher(Configuration &conf) : conf(conf) {
	}

	void launch();

};

template<typename Matrix, typename Strategy>
CrossValidationSolver<CGaussKernel, Matrix, Strategy>* ApplicationLauncher::createCrossValidator() {
	ifstream input(conf.dataFile.c_str());
	BaseSolverFactory<Matrix, Strategy> reader(
			input, conf.trainingParams, conf.stopCriterion, conf.multiclass, false);

	Timer timer(true);
	CrossValidationSolver<CGaussKernel, Matrix, Strategy> *solver
			= (CrossValidationSolver<CGaussKernel, Matrix, Strategy>*) reader.getCrossValidationSolver(
					conf.validation.innerFolds, conf.validation.outerFolds);
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();
	input.close();
	return solver;
}

template<typename Matrix, typename Strategy>
AbstractSolver<CGaussKernel, Matrix, Strategy>* ApplicationLauncher::createSolver() {
	ifstream input(conf.dataFile.c_str());
	BaseSolverFactory<Matrix, Strategy> reader(input, conf.trainingParams, conf.stopCriterion, conf.multiclass, false);

	Timer timer(true);
	AbstractSolver<CGaussKernel, Matrix, Strategy> *solver = reader.getSolver();
	timer.stop();

	logger << format("input reading time: %.2f[s]\n") % timer.getTimeElapsed();
	input.close();
	return solver;
}

template<typename Matrix, typename Strategy>
GridGaussianModelSelector<Matrix, Strategy>* ApplicationLauncher::createModelSelector() {
	GridGaussianModelSelector<Matrix, Strategy> *selector;
	if (conf.validation.modelSelection == GRID) {
		selector = new GridGaussianModelSelector<Matrix, Strategy>();
	} else {
		PatternFactory factory;
		selector = new PatternGaussianModelSelector<Matrix, Strategy>(factory.createCross());
	}
	return selector;
}

template<typename Matrix, typename Strategy>
void ApplicationLauncher::performModelSelection() {
	CrossValidationSolver<CGaussKernel, Matrix, Strategy> *solver
			= createCrossValidator<Matrix, Strategy>();

	Timer timer(true);
	GridGaussianModelSelector<Matrix, Strategy> *selector
			= createModelSelector<Matrix, Strategy>();
	ModelSelectionResults params = selector->selectParameters(*solver, conf.searchRange);
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%], C=%.4g, G=%.4g\n")
			% timer.getTimeElapsed() % (100.0 * params.bestResult.accuracy)
			% params.c % params.gamma;

	delete solver;
	delete selector;
}

template<typename Matrix, typename Strategy>
void ApplicationLauncher::performNestedCrossValidation() {
	CrossValidationSolver<CGaussKernel, Matrix, Strategy> *solver
			= createCrossValidator<Matrix, Strategy>();

	Timer timer(true);
	GridGaussianModelSelector<Matrix, Strategy> *selector
			= createModelSelector<Matrix, Strategy>();
	TestingResult res = selector->doNestedCrossValidation(*solver, conf.searchRange);
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * res.accuracy);

	delete solver;
	delete selector;
}

template<typename Matrix, typename Strategy>
void ApplicationLauncher::performCrossValidation() {
	CrossValidationSolver<CGaussKernel, Matrix, Strategy> *solver
			= createCrossValidator<Matrix, Strategy>();

	Timer timer(true);
	CGaussKernel param(conf.searchRange.gammaLow);
	solver->setKernelParams(conf.searchRange.cLow, param);
	TestingResult result = solver->doCrossValidation();
	timer.stop();

	logger << format("final result: time=%.2f[s], accuracy=%.2f[%%]\n")
			% timer.getTimeElapsed() % (100.0 * result.accuracy);

	delete solver;
}

template<typename Matrix, typename Strategy>
void ApplicationLauncher::performTraining() {
	AbstractSolver<CGaussKernel, Matrix, Strategy> *solver = createSolver<Matrix, Strategy>();

	Timer timer(true);
	CGaussKernel param(conf.searchRange.gammaLow);
	solver->setKernelParams(conf.searchRange.cLow, param);
	solver->train();
	Classifier<CGaussKernel, Matrix>* classifier = solver->getClassifier();
	timer.stop();

	Matrix *samples = solver->getSamples();
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
	delete classifier;
}

template<typename Matrix, ViolationCriterion Violation, GeneratorType Generator>
void ApplicationLauncher::run() {
	if (conf.validation.outerFolds > 1) {
		performNestedCrossValidation<Matrix, SolverStrategy<Violation, Generator> >();
	} else {
		if (conf.validation.innerFolds > 1) {
			if (conf.searchRange.cResolution > 1 || conf.searchRange.gammaResolution > 1) {
				performModelSelection<Matrix, SolverStrategy<Violation, Generator> >();
			} else {
				performCrossValidation<Matrix, SolverStrategy<Violation, Generator> >();
			}
		} else {
			performTraining<Matrix, SolverStrategy<Violation, Generator> >();
		}
	}
}

template<typename Matrix, ViolationCriterion Violation>
void ApplicationLauncher::selectGeneratorTypeAndRun() {
	switch (conf.randomization) {
	case PLAIN:
		run<Matrix, Violation, PLAIN>();
		break;
	case FAIR:
		run<Matrix, Violation, FAIR>();
		break;
	case DETERMINISTIC:
		run<Matrix, Violation, DETERMINISTIC>();
		break;
	default:
		throw invalid_configuration("unknown randomization type");
	}
}

template<typename Matrix>
void ApplicationLauncher::selectViolationCriterionAndRun() {
	switch (conf.optimizationProcedure) {
	case L1SVM:
		selectGeneratorTypeAndRun<Matrix, L1SVM>();
		break;
	default:
		throw invalid_configuration("unknown optimization criterion");
	}
}

#endif
