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

#ifndef SOLVER_H_
#define SOLVER_H_

#include <string>
#include <iostream>

#include <map>
#include <vector>

#include <boost/smart_ptr.hpp>

#include "kernel.h"
#include "classify.h"
#include "cache.h"
#include "stop.h"
#include "bias.h"
#include "params.h"
#include "../math/random.h"
#include "../math/average.h"

#define STARTING_EPSILON 1.0
#define EPSILON_SHRINKING_FACTOR 2.0

#define CACHE_USAGE_RATIO 0.1

using namespace std;
using boost::scoped_ptr;

template<typename Kernel, typename Matrix>
class Solver {

public:
	virtual ~Solver() {};

	virtual void setKernelParams(fvalue c, Kernel &params) = 0;
	virtual void train() = 0;
	virtual Classifier<Kernel, Matrix>* getClassifier() = 0;

};


template<typename Matrix>
class DataHolder {

public:
	virtual ~DataHolder() {};

	virtual Matrix* getSamples() = 0;
	virtual label_id* getLabels() = 0;
	virtual map<label_id, string>& getLabelNames() = 0;

};


template<typename Matrix>
class StateHolder: public DataHolder<Matrix> {

public:
	virtual ~StateHolder() {};

	virtual void setSwapListener(SwapListener *listener) = 0;
	virtual void swapSamples(sample_id u, sample_id v) = 0;
	virtual void shrink() = 0;
	virtual void releaseSupportVectors(fold_id *membership, fold_id fold) = 0;
	virtual void setCurrentSize(quantity size) = 0;
	virtual quantity getCurrentSize() = 0;
	virtual void reset() = 0;

	virtual Matrix* getSamples() = 0;
	virtual label_id* getLabels() = 0;
	virtual map<label_id, string>& getLabelNames() = 0;

	virtual quantity getSize() = 0;
	virtual quantity getSvNumber() = 0;

};


struct ViolatorSearch {

	sample_id violator;
	fvalue yo;

	ViolatorSearch(sample_id violator, fvalue yo) :
		violator(violator),
		yo(yo) {
	}

};

template<typename Kernel, typename Matrix, typename Strategy>
class AbstractSolver: public Solver<Kernel, Matrix>, public StateHolder<Matrix> {

	SwapListener *listener;

protected:
	TrainParams params;
	StopCriterionStrategy *stopStrategy;

	map<label_id, string> labelNames;

	quantity dimension;
	quantity size;
	quantity currentSize;

	Matrix *samples;
	label_id *labels;

	Strategy strategy;

	CachedKernelEvaluator<Kernel, Matrix, Strategy> *cache;

protected:
	ViolatorSearch findWorstViolator();

	virtual CachedKernelEvaluator<Kernel, Matrix, Strategy>* buildCache(
			fvalue c, Kernel &gparams);

	void trainForCache(CachedKernelEvaluator<Kernel, Matrix, Strategy> *cache);

	void refreshDistr();

public:
	AbstractSolver(map<label_id, string> labelNames, Matrix *samples,
			label_id *labels, TrainParams &params,
			StopCriterionStrategy *stopStrategy);
	virtual ~AbstractSolver();

	void setKernelParams(fvalue c, Kernel &params);
	virtual void train() = 0;
	Classifier<Kernel, Matrix>* getClassifier() = 0;

	void setSwapListener(SwapListener *listener);
	void swapSamples(sample_id u, sample_id v);
	void shrink();
	void releaseSupportVectors(fold_id *membership, fold_id fold);
	void setCurrentSize(quantity size);
	quantity getCurrentSize();
	void reset();

	Matrix* getSamples();
	label_id* getLabels();
	map<label_id, string>& getLabelNames();

	quantity getSize();
	virtual quantity getSvNumber();

	void reportStatistics();

};

template<typename Kernel, typename Matrix, typename Strategy>
AbstractSolver<Kernel, Matrix, Strategy>::AbstractSolver(
		map<label_id, string> labelNames, Matrix *samples,
		label_id *labels, TrainParams &params,
		StopCriterionStrategy *stopStrategy) :
		params(params),
		stopStrategy(stopStrategy),
		labelNames(labelNames),
		samples(samples),
		labels(labels),
		strategy(Strategy(params, (quantity) labelNames.size(), labels, (quantity) samples->height)) {
	size = (quantity) samples->height;
	currentSize = (quantity) samples->height;
	dimension = (quantity) samples->width;

	listener = NULL;
	cache = NULL;

	refreshDistr();
}

template<typename Kernel, typename Matrix, typename Strategy>
AbstractSolver<Kernel, Matrix, Strategy>::~AbstractSolver() {
	if (cache) {
		delete cache;
	}
	delete samples;
	delete [] labels;
	delete stopStrategy;
}

/*
 * Sets the kernel parameters for the current model. If the cache is not set,
 * initialize the cache as well.
 */
template<typename Kernel, typename Matrix, typename Strategy>
void AbstractSolver<Kernel, Matrix, Strategy>::setKernelParams(
		fvalue c, Kernel &gparams) {
	if (cache == NULL) {
		cache = buildCache(c, gparams);
		cache->setSwapListener(listener);
	} else {
		cache->setKernelParams(c, gparams);
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void AbstractSolver<Kernel, Matrix, Strategy>::reportStatistics() {
	cache->reportStatistics();
}

/*
 * Find the worst violator. Since the violators are stacked at the top, iterate only
 * over the non-support vectors to find the next worst violator.
 */
template<typename Kernel, typename Matrix, typename Strategy>
ViolatorSearch AbstractSolver<Kernel, Matrix, Strategy>::findWorstViolator() {
	ViolatorSearch worst_viol(INVALID_SAMPLE_ID, 0);
	fvalue min_val = 0;
	fvalue ksi = 0.0;
	quantity svnumber = cache->getSVNumber();
	for (sample_id i = svnumber; i < currentSize; i++) {
		ksi = cache->checkViolation(i);
		if (ksi < min_val) {
			worst_viol.violator = i;
			worst_viol.yo = ksi;
			min_val = ksi;
		}
	}
	return worst_viol;
}

/*
 * Begin training. 
 */
template<typename Kernel, typename Matrix, typename Strategy>
void AbstractSolver<Kernel, Matrix, Strategy>::trainForCache(
		CachedKernelEvaluator<Kernel, Matrix, Strategy> *cache) {
	ViolatorSearch viol(0, 0.0);
	fvalue C = cache->getC();
	fvalue margin = 0.1*C;
	quantity iter = 0;
	fvalue bias = 0.0;
	fvalue eta = 0.0;
	quantity max_iter = (quantity) ceil(0.5*currentSize);
	fvalue lambda = 0.0;

	do {
		iter += 1;
		eta = 2.0 / sqrt(iter);

		lambda = eta*C*cache->getLabel(viol.violator);
		cache->performUpdate(viol.violator, lambda);
		viol = findWorstViolator();
		cache->performSvUpdate(viol.violator);

	} while (iter < max_iter && viol.yo < margin);
}

template<typename Kernel, typename Matrix, typename Strategy>
CachedKernelEvaluator<Kernel, Matrix, Strategy>* AbstractSolver<Kernel, Matrix, Strategy>::buildCache(fvalue c, Kernel &gparams) {
	fvalue bias = (params.bias == NO) ? 0.0 : 1.0;
	RbfKernelEvaluator<GaussKernel, Matrix> *rbf = new RbfKernelEvaluator<GaussKernel, Matrix>(
			this->samples, this->labels, (quantity) labelNames.size(), bias, c, gparams);
	return new CachedKernelEvaluator<GaussKernel, Matrix, Strategy>(
			rbf, &strategy, size, params.cache.size, NULL);
}

template<typename Kernel, typename Matrix, typename Strategy>
Matrix* AbstractSolver<Kernel, Matrix, Strategy>::getSamples() {
	return samples;
}

template<typename Kernel, typename Matrix, typename Strategy>
label_id* AbstractSolver<Kernel, Matrix, Strategy>::getLabels() {
	return labels;
}

template<typename Kernel, typename Matrix, typename Strategy>
map<label_id, string>& AbstractSolver<Kernel, Matrix, Strategy>::getLabelNames() {
	return labelNames;
}

template<typename Kernel, typename Matrix, typename Strategy>
void AbstractSolver<Kernel, Matrix, Strategy>::refreshDistr() {
	this->strategy.resetGenerator(labels, currentSize);
}

template<typename Kernel, typename Matrix, typename Strategy>
void AbstractSolver<Kernel, Matrix, Strategy>::setSwapListener(
		SwapListener *listener) {
	this->listener = listener;
	if (cache) {
		cache->setSwapListener(listener);
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void AbstractSolver<Kernel, Matrix, Strategy>::swapSamples(
		sample_id u, sample_id v) {
	cache->swapSamples(u, v);
}

template<typename Kernel, typename Matrix, typename Strategy>
void AbstractSolver<Kernel, Matrix, Strategy>::reset() {
	cache->reset();
}

template<typename Kernel, typename Matrix, typename Strategy>
void AbstractSolver<Kernel, Matrix, Strategy>::shrink() {
	cache->shrink();
}

template<typename Kernel, typename Matrix, typename Strategy>
void AbstractSolver<Kernel, Matrix, Strategy>::releaseSupportVectors(
		fold_id *membership, fold_id fold) {
	cache->releaseSupportVectors(membership, fold);
}

template<typename Kernel, typename Matrix, typename Strategy>
void AbstractSolver<Kernel, Matrix, Strategy>::setCurrentSize(quantity size) {
	currentSize = size;
	refreshDistr();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline quantity AbstractSolver<Kernel, Matrix, Strategy>::getCurrentSize() {
	return currentSize;
}

template<typename Kernel, typename Matrix, typename Strategy>
quantity AbstractSolver<Kernel, Matrix, Strategy>::getSize() {
	return size;
}

template<typename Kernel, typename Matrix, typename Strategy>
quantity AbstractSolver<Kernel, Matrix, Strategy>::getSvNumber() {
	return this->cache->getSVNumber();
}

#endif
