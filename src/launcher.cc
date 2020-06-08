
#include "launcher.h"

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
	PatternFactory factory;
	selector = new PatternGaussianModelSelector(factory.createCross());
	return selector;
}

Classifier* ApplicationLauncher::performModelSelection() {
	CrossValidationSolver *solver = createCrossValidator();

	Timer timer(true);
	GridGaussianModelSelector *selector = createModelSelector();
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

	// create test cases if user specified
	if(conf.createTestCases){
		classifier->saveClassifier(conf.testName);
	}
	
}
