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

#ifndef SOLVER_FACTORY_H_
#define SOLVER_FACTORY_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include <set>
#include <map>
#include <list>
#include <vector>

#include "../svm/strategy.h"
#include "dataset.h"
#include "../logging/log.h"
#include "../svm/pairwise_solver.h"
#include "../svm/validation.h"
#include "../svm/strategy.h"
#include "../feature/feature.h"

using namespace std;

//TODO: since i deleted the universal solver class, i need to make sure that the run-options reflect that it doesnt exist anymore
enum StopCriterion {
	YOC
};

enum MulticlassApproach {
	PAIRWISE
};

class FeatureMatrixBuilder {

public:
	sfmatrix* getFeatureMatrix(list<map<feature_id, fvalue> >& features, map<feature_id, feature_id>& mappings);
};

class BaseSolverFactory {

private:
	istream& input;

	TrainParams params;
	StopCriterion strategy;
	MulticlassApproach multiclass;

	bool reduceDim;

	FeatureMatrixBuilder *matrixBuilder;

	map<feature_id, feature_id> findOptimalFeatureMappings(list<map<feature_id, fvalue> >& features);
	label_id* getLabelVector(list<label_id>& labels);
	StopCriterionStrategy* getStopCriterion();

  	sfmatrix* preprocess(sfmatrix *x, label_id *y);

	AbstractSolver* createSolver(MulticlassApproach type, map<label_id,string> labels, sfmatrix* x, label_id* y, TrainParams& params, StopCriterionStrategy* strategy);

public:
	BaseSolverFactory(istream& input, TrainParams params = TrainParams(), StopCriterion strategy = YOC);
	virtual ~BaseSolverFactory();

	AbstractSolver* getSolver();
	CrossValidationSolver* getCrossValidationSolver(quantity innerFolds, quantity outerFolds);

};


#endif
