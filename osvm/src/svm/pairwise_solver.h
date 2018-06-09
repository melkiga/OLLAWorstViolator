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
template<typename Matrix>
class PairwiseClassifier: public Classifier<Matrix> {

	RbfKernelEvaluator<Matrix>* evaluator;
	PairwiseTrainingResult* state;
	fvector *buffer;

	vector<quantity> votes;
	vector<fvalue> evidence;

protected:
	fvalue getDecisionForModel(sample_id sample,
			PairwiseTrainingModel* model, fvector* buffer);
	fvalue convertDecisionToEvidence(fvalue decision);

public:
	PairwiseClassifier(RbfKernelEvaluator<Matrix> *evaluator,
			PairwiseTrainingResult* state, fvector *buffer);
	virtual ~PairwiseClassifier();

	virtual label_id classify(sample_id sample);
	virtual quantity getSvNumber();

};

template<typename Matrix>
PairwiseClassifier<Matrix>::PairwiseClassifier(
		RbfKernelEvaluator<Matrix> *evaluator,
		PairwiseTrainingResult* state, fvector *buffer) :
		evaluator(evaluator),
		state(state),
		buffer(buffer),
		votes(state->totalLabelCount),
		evidence(state->totalLabelCount){
}

template<typename Matrix>
PairwiseClassifier<Matrix>::~PairwiseClassifier() {
}

template<typename Matrix>
label_id PairwiseClassifier<Matrix>::classify(sample_id sample) {
	fill(votes.begin(), votes.end(), 0);
	fill(evidence.begin(), evidence.end(), 0.0);

	evaluator->evalKernel(sample, 0, state->maxSVCount, buffer);

	vector<PairwiseTrainingModel>::iterator it;
	for (it = state->models.begin(); it != state->models.end(); it++) {
		PairwiseTrainingModel* result = &(*it);
		fvalue dec = getDecisionForModel(sample, result, buffer);
		label_id label = dec > 0
				? result->trainingLabels.first
				: result->trainingLabels.second;
		votes[label]++;
		fvalue evidValue = convertDecisionToEvidence(dec);
		evidence[result->trainingLabels.first] += evidValue;
		evidence[result->trainingLabels.second] += evidValue;
	}

	label_id maxLabelId = 0;
	quantity maxVotes = 0;
	quantity maxEvidence = (quantity) 0.0;
	for (label_id i = 0; i < state->totalLabelCount; i++) {
		if (votes[i] > maxVotes
			|| (votes[i] == maxVotes && evidence[i] > maxEvidence)) {
			maxLabelId = i;
			maxVotes = votes[i];
			maxEvidence = (quantity) evidence[i];
		}
	}

	return maxLabelId;
}

template<typename Matrix>
fvalue PairwiseClassifier<Matrix>::getDecisionForModel(sample_id sample,
		PairwiseTrainingModel* model, fvector* buffer) {
	fvalue dec = model->bias;
	fvalue* kernels = buffer->data;
	for (sample_id i = 0; i < model->size; i++) {
		dec += model->yalphas[i] * kernels[model->samples[i]];
	}
	return dec;
}

template<typename Matrix>
inline fvalue PairwiseClassifier<Matrix>::convertDecisionToEvidence(
		fvalue decision) {
	return decision;
}

template<typename Matrix>
quantity PairwiseClassifier<Matrix>::getSvNumber() {
	return state->maxSVCount;
}


/**
 * Pairwise solver performs SVM training by generating SVM state for all
 * two-element combinations of the class trainingLabels.
 */
template<typename Matrix, typename Strategy>
class PairwiseSolver: public AbstractSolver<Matrix, Strategy> {

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
	CachedKernelEvaluator<Matrix, Strategy>* buildCache(
			fvalue c, CGaussKernel &gparams);

public:
	PairwiseSolver(map<label_id, string> labelNames, Matrix *samples,
			label_id *labels, TrainParams &params,
			StopCriterionStrategy *stopStrategy);
	virtual ~PairwiseSolver();

	void train();
	Classifier<Matrix>* getClassifier();

	quantity getSvNumber();

};


template<typename Matrix, typename Strategy>
PairwiseSolver<Matrix, Strategy>::PairwiseSolver(
		map<label_id, string> labelNames, Matrix *samples,
		label_id *labels, TrainParams &params,
		StopCriterionStrategy *stopStrategy) :
		AbstractSolver<Matrix, Strategy>(labelNames,
				samples, labels, params, stopStrategy),
		state(PairwiseTrainingResult()) {
	label_id maxLabel = (label_id) labelNames.size();
	vector<quantity> classSizes(maxLabel, 0);
	for (sample_id sample = 0; sample < this->size; sample++) {
		classSizes[this->labels[sample]]++;
	}

	// sort
	vector<pair<label_id, quantity> > sizes(maxLabel);
	for (label_id label = 0; label < maxLabel; label++) {
		sizes[label] = pair<label_id, quantity>(label, classSizes[label]);
	}
	sort(sizes.begin(), sizes.end(), PairValueComparator<label_id, quantity>());

	vector<pair<label_id, quantity> >::iterator it1;
	for (it1 = sizes.begin(); it1 < sizes.end(); it1++) {
		vector<pair<label_id, quantity> >::iterator it2;
		for (it2 = it1 + 1; it2 < sizes.end(); it2++) {
			pair<label_id, label_id> labels(it1->first, it2->first);
			quantity size = it1->second + it2->second;
			state.models.push_back(PairwiseTrainingModel(labels, size));
		}
	}
	state.totalLabelCount = maxLabel;
}

template<typename Matrix, typename Strategy>
PairwiseSolver<Matrix, Strategy>::~PairwiseSolver() {
}

template<typename Matrix, typename Strategy>
Classifier<Matrix>* PairwiseSolver<Matrix, Strategy>::getClassifier() {
	return new PairwiseClassifier<Matrix>(this->cache->getEvaluator(),
			&state, this->cache->getBuffer());
}

template<typename Matrix, typename Strategy>
void PairwiseSolver<Matrix, Strategy>::train() {
	RbfKernelEvaluator<Matrix>* evaluator = this->cache->getEvaluator();

	quantity totalSize = this->currentSize;
	vector<PairwiseTrainingModel>::iterator it;
	for (it = state.models.begin(); it != state.models.end(); it++) {
		pair<label_id, label_id> trainPair = it->trainingLabels;
		quantity size = reorderSamples(this->labels, totalSize, trainPair);
		this->cache->setLabel(trainPair);
		this->setCurrentSize(size);
		this->reset();
		this->trainForCache(this->cache);

		it->yalphas = this->cache->getAlphas();
		it->samples = this->cache->getBackwardOrder();
		it->bias = this->cache->getBias();
		it->size = this->cache->getSVNumber() - 1;
	}

	id freeOffset = 0;
	vector<sample_id>& mapping = this->cache->getForwardOrder();
	for (it = state.models.begin(); it != state.models.end(); it++) {
		for (id i = 0; i < it->size; i++) {
			id realOffset = mapping[it->samples[i]];
			if (realOffset >= freeOffset) {
				this->swapSamples(realOffset, freeOffset);
				realOffset = freeOffset++;
			}
			it->samples[i] = realOffset;
		}
	}
	state.maxSVCount = freeOffset;

	this->setCurrentSize(totalSize);
}

template<typename Matrix, typename Strategy>
quantity PairwiseSolver<Matrix, Strategy>::reorderSamples(
		label_id *labels, quantity size, pair<label_id, label_id>& labelPair) {
	label_id first = labelPair.first;
	label_id second = labelPair.second;
	id train = 0;
	id test = size - 1;
	while (train <= test) {
		while (train < size && (labels[train] == first || labels[train] == second)) {
			train++;
		}
		while (test >= 0 && (labels[test] != first && labels[test] != second)) {
			test--;
		}
		if (train < test) {
			this->swapSamples(train++, test--);
		}
	}
	return train;
}

template<typename Matrix, typename Strategy>
CachedKernelEvaluator<Matrix, Strategy>* PairwiseSolver<Matrix, Strategy>::buildCache(
		fvalue c, CGaussKernel &gparams) {
	fvalue bias = (this->params.bias == NO) ? 0.0 : 1.0;
	RbfKernelEvaluator<Matrix> *rbf = new RbfKernelEvaluator<Matrix>(
			this->samples, this->labels, 2, bias, c, gparams, this->params.epochs, this->params.margin);
	return new CachedKernelEvaluator<Matrix, Strategy>(
			rbf, &this->strategy, this->size, this->params.cache.size, NULL);
}

template<typename Matrix, typename Strategy>
quantity PairwiseSolver<Matrix, Strategy>::getSvNumber() {
	return state.maxSVCount;
}

#endif
