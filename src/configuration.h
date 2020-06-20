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

#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <iostream>

#include <boost/program_options.hpp>

#include "data/solver_factory.h"
#include "svm/kernel.h"
#include "svm/strategy.h"
#include "time/timer.h"
#include "model/selection.h"
#include "logging/log.h"
#include "svm/stop.h"

using namespace std;
using namespace boost::program_options;

#define PACKAGE "osvm"

#define PR_HELP "help,h"
#define PR_C_LOW "c-low,c"
#define PR_C_HIGH "c-high,C"
#define PR_G_LOW "gamma-low,g"
#define PR_G_HIGH "gamma-high,G"
#define PR_RES "resolution,r"
#define PR_OUTER_FLD "outer-folds,o"
#define PR_INNER_FLD "inner-folds,i"
#define PR_CREATE_TESTS "create-test-cases,d"
#define PR_INPUT "input,I"
#define PR_CACHE_SIZE "cache-size,S"
#define PR_BIAS_CALCULATION "bias,b"
#define PR_EPOCH "epochs,P"
#define PR_MARGIN "margin,M"
#define PR_TEST_NAME "test-name,t"

#define PR_KEY_HELP "help"
#define PR_KEY_C_LOW "c-low"
#define PR_KEY_C_HIGH "c-high"
#define PR_KEY_G_LOW "gamma-low"
#define PR_KEY_G_HIGH "gamma-high"
#define PR_KEY_RES "resolution"
#define PR_KEY_OUTER_FLD "outer-folds"
#define PR_KEY_INNER_FLD "inner-folds"
#define PR_KEY_CREATE_TESTS "create-test-cases"
#define PR_KEY_INPUT "input"
#define PR_KEY_CACHE_SIZE "cache-size"
#define PR_KEY_BIAS "bias"
#define PR_KEY_EPOCH "epochs"
#define PR_KEY_MARGIN "margin"
#define PR_KEY_TEST_NAME "test-name"

#define BIAS_CALCULATION_NO "nobias"
#define BIAS_CALCULATION_YES "yesbias"

class invalid_configuration: public exception {

	string message;

public:
	invalid_configuration(string message);
	virtual ~invalid_configuration() throw();

	virtual const char* what() const throw();

};

enum MatrixType {
	SPARSE,
	DENSE
};

enum ModelSelectionType {
	GRID,
	PATTERN
};

struct Configuration {
	string dataFile;
	bool createTestCases;
	string testName;

	SearchRange searchRange;
	TrainParams trainingParams;

	struct CrossValidationParams {
		quantity innerFolds;
		quantity outerFolds;
		ModelSelectionType modelSelection;
	} validation;

	MatrixType matrixType;
	StopCriterion stopCriterion;
	MulticlassApproach multiclass;
};

class ParametersParser {
	
	variables_map& vars;

public:
	ParametersParser(variables_map& vars);
	
	Configuration getConfiguration();
};


#endif
