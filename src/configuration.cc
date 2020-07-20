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

#include "configuration.h"

invalid_configuration::invalid_configuration(string message) : message(message) {
}

invalid_configuration::~invalid_configuration() throw () {
}

const char* invalid_configuration::what() const throw() {
	return message.c_str();
}

ParametersParser::ParametersParser(bopt::variables_map& vars) :
		vars(vars) {
}

Configuration ParametersParser::getConfiguration() {
	Configuration conf;

	if (vars.count(PR_KEY_INPUT)) {
		conf.dataFile = vars[PR_KEY_INPUT].as<string>();
		ifstream dataStream(conf.dataFile.c_str());
		if (!dataStream) {
			string msg = (format("input file '%s' does not exist") % conf.dataFile).str();
			throw invalid_configuration(msg);
		}
	} else {
		throw invalid_configuration("input file not specified");
	}

  // SVM cross validation parameter setting
	SearchRange range;
	range.cResolution = vars[PR_KEY_RES].as<int>();
	range.cLow = vars[PR_KEY_C_LOW].as<fvalue>();
	range.cHigh = vars[PR_KEY_C_HIGH].as<fvalue>();
	range.gammaResolution = vars[PR_KEY_RES].as<int>();
	range.gammaLow = vars[PR_KEY_G_LOW].as<fvalue>();
	range.gammaHigh = vars[PR_KEY_G_HIGH].as<fvalue>();
	conf.searchRange = range;

	quantity drawNumber = 600;
	quantity cacheSize = vars[PR_KEY_CACHE_SIZE].as<int>();

  // Whether to use bias in our SVM model or not. The default is yes.
	BiasType bias = YES;
	string biasEvaluation = vars[PR_KEY_BIAS].as<string>();
	if (BIAS_CALCULATION_NO == biasEvaluation) {
		bias = NO;
	} else if (BIAS_CALCULATION_YES == biasEvaluation) {
		bias = YES;
	} else {
		throw invalid_configuration("invalid bias evaluation strategy: " + biasEvaluation);
	}

	fvalue epochs = vars[PR_KEY_EPOCH].as<fvalue>();
	fvalue margin = vars[PR_KEY_MARGIN].as<fvalue>();

	TrainParams params;
	params.bias = bias;
	params.drawNumber = drawNumber;
	params.cache.size = cacheSize;
	params.epochs = epochs;
	params.margin = margin;
	conf.trainingParams = params;

  // cross-validation inner and outer folds
	conf.validation.innerFolds = vars[PR_KEY_INNER_FLD].as<int>();
	conf.validation.outerFolds = vars[PR_KEY_OUTER_FLD].as<int>();
  // TODO: eventually we don't need this in the configuration because we always use pairwise
	conf.multiclass = PAIRWISE;

  // TODO: we only really use pattern search
	conf.validation.modelSelection = PATTERN;

	conf.createTestCases = vars[PR_KEY_CREATE_TESTS].as<bool>();
	if(conf.createTestCases){
		conf.testName = vars[PR_KEY_TEST_NAME].as<string>();
	}

	return conf;
}
