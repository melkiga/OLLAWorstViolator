#ifndef LAUCHER_H_
#define LAUCHER_H_

#include "configuration.h"

class ApplicationLauncher {

	Configuration &conf;

protected:
	CrossValidationSolver* createCrossValidator();

	AbstractSolver* createSolver();

	GridGaussianModelSelector* createModelSelector();

	Classifier* performTraining();

	Classifier* performCrossValidation();

	Classifier* performModelSelection();

	Classifier* performNestedCrossValidation();

public:
	ApplicationLauncher(Configuration &conf) : conf(conf) {
	}

  void run();

};

#endif
