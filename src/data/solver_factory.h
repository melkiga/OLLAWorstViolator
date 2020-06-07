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


BaseSolverFactory::BaseSolverFactory(istream& input, TrainParams params, StopCriterion strategy) :
		input(input),
		params(params),
		strategy(strategy),
		matrixBuilder(new FeatureMatrixBuilder()) {
}


BaseSolverFactory::~BaseSolverFactory() {
	delete matrixBuilder;
}


AbstractSolver* BaseSolverFactory::createSolver(MulticlassApproach type, map<label_id, string> labels, sfmatrix* x, label_id* y, TrainParams& params, StopCriterionStrategy* strategy) {
	AbstractSolver *solver = NULL;
	solver = new PairwiseSolver(labels, x, y, params, strategy);
	return solver;
}



AbstractSolver* BaseSolverFactory::getSolver() {
	SparseFormatDataSetFactory dataSetFactory(input);
	DataSet dataSet = dataSetFactory.createDataSet();

	map<feature_id, feature_id> mappings = findOptimalFeatureMappings(dataSet.features);
	sfmatrix *x = matrixBuilder->getFeatureMatrix(dataSet.features, mappings);
	label_id *y = getLabelVector(dataSet.labels);

	x = preprocess(x, y);

	StopCriterionStrategy *strategy = getStopCriterion();

	return createSolver(multiclass, dataSet.labelNames, x, y, params, strategy);
}


CrossValidationSolver* BaseSolverFactory::getCrossValidationSolver(quantity innerFolds, quantity outerFolds) {
	AbstractSolver *solver = getSolver();
	return new CrossValidationSolver(solver, innerFolds, outerFolds);
}


map<feature_id, feature_id> BaseSolverFactory::findOptimalFeatureMappings(list<map<feature_id, fvalue> >& features) {
	quantity dimension = 0;
	map<feature_id, fvalue> fmax;
	map<feature_id, fvalue> fmin;

	list<map<feature_id, fvalue> >::iterator lit;
	for (lit = features.begin(); lit != features.end(); lit++) {
		map<feature_id, fvalue>::iterator mit;
		for (mit = lit->begin(); mit != lit->end(); mit++) {
			dimension = max((quantity) mit->first, dimension);
			fmax[mit->first] = max(fmax[mit->first], mit->second);
			fmin[mit->first] = min(fmin[mit->first], mit->second);
		}
	}

	feature_id available = 0;
	map<feature_id, feature_id> mappings;
	for (feature_id feat = 0; feat <= dimension; feat++) {
		if (fmax[feat] == fmin[feat]) {
			for (lit = features.begin(); lit != features.end(); lit++) {
				lit->erase(feat);
			}
		} else {
			mappings[feat] = available++;
		}
	}

	quantity featureNum = 0;
	for (lit = features.begin(); lit != features.end(); lit++) {
		featureNum += (quantity) lit->size();
	}

	double density = 100.0 * featureNum / (features.size() * mappings.size());
	logger << format("data density: %.2f[%%]") % density << endl;
	return mappings;
}


label_id* BaseSolverFactory::getLabelVector(list<label_id>& labels) {
	label_id *dataLabels = new label_id[labels.size()];
	quantity id = 0;

	list<label_id>::iterator lit;
	for (lit = labels.begin(); lit != labels.end(); lit++) {
		dataLabels[id++] = *lit;
	}
	return dataLabels;
}


StopCriterionStrategy* BaseSolverFactory::getStopCriterion() {
	return new L1SVMStopStrategy();
}


sfmatrix* BaseSolverFactory::preprocess(sfmatrix *x, label_id *y) {
	FeatureProcessor proc;
	proc.normalize(x);
	proc.randomize(x, y);
	return x;
}

#endif
