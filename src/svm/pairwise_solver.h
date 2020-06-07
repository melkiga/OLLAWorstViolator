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

#ifndef SOLVER_PAIRWISE_H_
#define SOLVER_PAIRWISE_H_

#include "solver.h"
#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>

// Short alias for this namespace
namespace pt = boost::property_tree;
struct PairwiseTrainingModel {

	pair<label_id, label_id> trainingLabels;
	vector<fvalue> yalphas;
	fvalue bias;
	vector<sample_id> samples;
	quantity size;

	PairwiseTrainingModel(pair<label_id, label_id>& trainingLabels, quantity size) :
			trainingLabels(trainingLabels),
			yalphas(vector<fvalue>(size, 0.0)),
			bias(0),
			samples(vector<sample_id>(size, INVALID_SAMPLE_ID)),
			size(0) {
	}

	void clear() {
		yalphas.clear();
		bias = 0;
		samples.clear();
	}

};

struct PairwiseTrainingResult {

	vector<PairwiseTrainingModel> models;
	quantity maxSVCount;
	quantity totalLabelCount;

};

/**
 * Pairwise classifier perform classification based on SVM models created by
 * pairwise solver.
 */
class PairwiseClassifier: public Classifier {

	RbfKernelEvaluator* evaluator;
	PairwiseTrainingResult* state;
	fvector *buffer;

	vector<quantity> votes;
	vector<fvalue> evidence;

protected:
	fvalue getDecisionForModel(sample_id sample, 
    PairwiseTrainingModel* model, fvector* buffer);
	fvalue convertDecisionToEvidence(fvalue decision);

public:
	PairwiseClassifier(RbfKernelEvaluator *evaluator, 
    PairwiseTrainingResult* state, fvector *buffer);
	virtual ~PairwiseClassifier();

	virtual label_id classify(sample_id sample);
	virtual quantity getSvNumber();
	virtual void saveClassifier();

  PairwiseTrainingResult* getState() { return state; }

};

/**
 * Pairwise solver performs SVM training by generating SVM state for all
 * two-element combinations of the class trainingLabels.
 */
class PairwiseSolver: public AbstractSolver {

	template<typename K, typename V>
	struct PairValueComparator {

		bool operator()(pair<K,V> pair1, pair<K,V> pair2) {
			return pair1.second > pair2.second;
		}

	};

	PairwiseTrainingResult state;

	quantity reorderSamples(label_id *labels, quantity size,
			pair<label_id, label_id>& labelPair);

protected:
	CachedKernelEvaluator* buildCache(fvalue c, CGaussKernel &gparams);

public:
	PairwiseSolver(map<label_id, string> labelNames, sfmatrix *samples,
			label_id *labels, TrainParams &params,
			StopCriterionStrategy *stopStrategy);
	virtual ~PairwiseSolver();

	void train();
	Classifier* getClassifier();

	quantity getSvNumber();

};

#endif
