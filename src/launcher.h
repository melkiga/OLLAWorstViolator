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

  	void run(pt::ptree& model);

  	template<class T, class... Args>
	shared_ptr<T> make_shared(Args&&... args)
	{
		return shared_ptr<T>( new T( std::forward<Args>( args )... ) );
	}

};

#endif
