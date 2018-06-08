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

#define STARTING_EPSILON 1.0
#define EPSILON_SHRINKING_FACTOR 2.0

#define CACHE_USAGE_RATIO 0.1

using namespace std;
using boost::scoped_ptr;

template<typename Matrix>
class Solver {

public:
	virtual ~Solver() {};

	virtual void setKernelParams(fvalue c, CGaussKernel &params) = 0;
	virtual void train() = 0;
	virtual Classifier<Matrix>* getClassifier() = 0;

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
	virtual void setCurrentSize(quantity size) = 0;
	virtual quantity getCurrentSize() = 0;
	virtual void reset() = 0;

	virtual Matrix* getSamples() = 0;
	virtual label_id* getLabels() = 0;
	virtual map<label_id, string>& getLabelNames() = 0;

	virtual quantity getSize() = 0;
	virtual quantity getSvNumber() = 0;

};


template<typename Matrix, typename Strategy>
class AbstractSolver: public Solver<Matrix>, public StateHolder<Matrix> {

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

	CachedKernelEvaluator<Matrix, Strategy> *cache;

protected:
	virtual CachedKernelEvaluator<Matrix, Strategy>* buildCache(fvalue c, CGaussKernel &gparams);
	void trainForCache(CachedKernelEvaluator<Matrix, Strategy> *cache);
	void refreshDistr();

public:
	AbstractSolver(map<label_id, string> labelNames, Matrix *samples, label_id *labels, TrainParams &params, StopCriterionStrategy *stopStrategy);
	virtual ~AbstractSolver();

	void setKernelParams(fvalue c, CGaussKernel &params);
	virtual void train() = 0;
	Classifier<Matrix>* getClassifier() = 0;

	void setSwapListener(SwapListener *listener);
	void swapSamples(sample_id u, sample_id v);
	void setCurrentSize(quantity size);
	quantity getCurrentSize();
	void reset();

	Matrix* getSamples();
	label_id* getLabels();
	map<label_id, string>& getLabelNames();

	quantity getSize();
	virtual quantity getSvNumber();
};

template<typename Matrix, typename Strategy>
AbstractSolver<Matrix, Strategy>::AbstractSolver(
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

template<typename Matrix, typename Strategy>
AbstractSolver<Matrix, Strategy>::~AbstractSolver() {
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
template<typename Matrix, typename Strategy>
void AbstractSolver<Matrix, Strategy>::setKernelParams(
		fvalue c, CGaussKernel &gparams) {
	if (cache == NULL) {
		cache = buildCache(c, gparams);
		cache->setSwapListener(listener);
	} else {
		cache->setKernelParams(c, gparams);
	}
}

/*
 * Training procedure for OLLAWV. This is basically the SGD procedure. First, we calculate the learning rate.
 * Next, we get the gradient for the alphas and the bias. We then update the model, find the next worst violator
 * with respect to the current decision function output. Finally, we 'bottom stack' the current worst violator (support vector)
 * replacing it with the corresponding non-support vector.
 */
template<typename Matrix, typename Strategy>
void AbstractSolver<Matrix, Strategy>::trainForCache(CachedKernelEvaluator<Matrix, Strategy> *cache) 
{
	CWorstViolator worstViolator(0, 0.0);
	fvalue svmPenaltyParameterC = cache->getC();
	fvalue useBias = cache->getBetta(); //TODO: change this to a bool
	fvalue margin = cache->getMargin()*svmPenaltyParameterC;
	quantity currentIteration = 0;
	fvalue learningRate = 0.0;
	quantity maxNumberOfIterations = (quantity) ceil(cache->getEpochs()*currentSize);
	fvalue alphasGradient = 0.0;
	fvalue biasGradient = 0.0;

	do {
		currentIteration += 1;
		learningRate = 2.0 / sqrt(currentIteration);

		alphasGradient = learningRate * svmPenaltyParameterC * cache->getLabel(worstViolator.m_violatorID);
		biasGradient = (alphasGradient * useBias) / currentSize;
		cache->performSGDUpdate(worstViolator.m_violatorID, alphasGradient, biasGradient);
		worstViolator = cache->findWorstViolator();
		cache->performSvUpdate(worstViolator.m_violatorID);

	} while (currentIteration < maxNumberOfIterations && worstViolator.m_error < margin);
}

template<typename Matrix, typename Strategy>
CachedKernelEvaluator<Matrix, Strategy>* AbstractSolver<Matrix, Strategy>::buildCache(fvalue c, CGaussKernel &gparams) {
	fvalue bias = (params.bias == NO) ? 0.0 : 1.0;
	RbfKernelEvaluator<Matrix> *rbf = new RbfKernelEvaluator<Matrix>(
			this->samples, this->labels, (quantity) labelNames.size(), bias, c, gparams, params.epochs, params.margin);
	return new CachedKernelEvaluator<Matrix, Strategy>(
			rbf, &strategy, size, params.cache.size, NULL);
}

template<typename Matrix, typename Strategy>
Matrix* AbstractSolver<Matrix, Strategy>::getSamples() {
	return samples;
}

template<typename Matrix, typename Strategy>
label_id* AbstractSolver<Matrix, Strategy>::getLabels() {
	return labels;
}

template<typename Matrix, typename Strategy>
map<label_id, string>& AbstractSolver<Matrix, Strategy>::getLabelNames() {
	return labelNames;
}

template<typename Matrix, typename Strategy>
void AbstractSolver<Matrix, Strategy>::refreshDistr() {
	this->strategy.resetGenerator(labels, currentSize);
}

template<typename Matrix, typename Strategy>
void AbstractSolver<Matrix, Strategy>::setSwapListener(
		SwapListener *listener) {
	this->listener = listener;
	if (cache) {
		cache->setSwapListener(listener);
	}
}

template<typename Matrix, typename Strategy>
void AbstractSolver<Matrix, Strategy>::swapSamples(
		sample_id u, sample_id v) {
	cache->swapSamples(u, v);
}

template<typename Matrix, typename Strategy>
void AbstractSolver<Matrix, Strategy>::reset() {
	cache->reset();
}

template<typename Matrix, typename Strategy>
void AbstractSolver<Matrix, Strategy>::setCurrentSize(quantity size) {
	currentSize = size;
	cache->setCurrentSize(size);
	refreshDistr();
}

template<typename Matrix, typename Strategy>
inline quantity AbstractSolver<Matrix, Strategy>::getCurrentSize() {
	return currentSize;
}

template<typename Matrix, typename Strategy>
quantity AbstractSolver<Matrix, Strategy>::getSize() {
	return size;
}

template<typename Matrix, typename Strategy>
quantity AbstractSolver<Matrix, Strategy>::getSvNumber() {
	return this->cache->getSVNumber();
}

#endif
