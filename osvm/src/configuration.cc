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


ParametersParser::ParametersParser(variables_map& vars) :
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

	SearchRange range;
	range.cResolution = vars[PR_KEY_RES].as<int>();
	range.cLow = vars[PR_KEY_C_LOW].as<fvalue>();
	range.cHigh = vars[PR_KEY_C_HIGH].as<fvalue>();
	range.gammaResolution = vars[PR_KEY_RES].as<int>();
	range.gammaLow = vars[PR_KEY_G_LOW].as<fvalue>();
	range.gammaHigh = vars[PR_KEY_G_HIGH].as<fvalue>();
	conf.searchRange = range;

	quantity drawNumber = vars[PR_KEY_DRAW_NUM].as<int>();
	quantity cacheSize = vars[PR_KEY_CACHE_SIZE].as<int>();
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

	string stopStr = vars[PR_KEY_STOP_CRIT].as<string>();
	StopCriterion stop;
	if (STOP_CRIT_YOC == stopStr) {
		stop = YOC;
	} else {
		throw invalid_configuration("invalid stopping criterion: " + stopStr);
	}
	conf.stopCriterion = stop;

	conf.validation.innerFolds = vars[PR_KEY_INNER_FLD].as<int>();
	conf.validation.outerFolds = vars[PR_KEY_OUTER_FLD].as<int>();

	string matrix = vars[PR_KEY_MATRIX_TYPE].as<string>();
	if (MAT_TYPE_SPARSE == matrix) {
		conf.matrixType = SPARSE;
	} else if (MAT_TYPE_DENSE == matrix) {
		conf.matrixType = DENSE;
	} else {
		string msg = (format("invalid matrix type: '%s'") % matrix).str();
		throw invalid_configuration(msg);
	}

	string random = vars[PR_KEY_ID_RANDOMIZER].as<string>();
	if (ID_RANDOMIZER_PLAIN == random) {
		conf.randomization = PLAIN;
	} else if (ID_RANDOMIZER_FAIR == random) {
		conf.randomization = FAIR;
	} else if (ID_RANDOMIZER_DETERM == random) {
		conf.randomization = DETERMINISTIC;
	} else {
		string msg = (format("invalid randomization type: '%s'") % random).str();
		throw invalid_configuration(msg);
	}

	string multiclass = vars[PR_KEY_MULTICLASS].as<string>();
	if (MULTICLASS_PAIRWISE == multiclass) {
		conf.multiclass = PAIRWISE;
	} else {
		string msg = (format("invalid multiclass approach: '%s'") % multiclass).str();
		throw invalid_configuration(msg);
	}

	string modelSelection = vars[PR_KEY_SEL_TYPE].as<string>();
	if (SEL_TYPE_GRID == modelSelection) {
		conf.validation.modelSelection = GRID;
	} else if (SEL_TYPE_PATTERN == modelSelection) {
		conf.validation.modelSelection = PATTERN;
	} else {
		string msg = (format("invalid model selection type: '%s'") % modelSelection).str();
		throw invalid_configuration(msg);
	}

	return conf;
}
