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

#include "strategy.h"
#include "kernel.h"
#include "classify.h"
#include "pairwise_solver.h"
#include "cache.h"
#include "stop.h"
#include "params.h"
#include "../math/random.h"

#define STARTING_EPSILON 1.0
#define EPSILON_SHRINKING_FACTOR 2.0

#define CACHE_USAGE_RATIO 0.1

using namespace std;
using boost::scoped_ptr;

class Solver {

public:
	virtual ~Solver() {};

	virtual void setKernelParams(fvalue c, CGaussKernel &params) = 0;
	virtual void train() = 0;
	virtual PairwiseClassifier getClassifier() = 0;

};


class DataHolder {

public:
	virtual ~DataHolder() {};

	virtual sfmatrix* getSamples() = 0;
	virtual label_id* getLabels() = 0;
	virtual map<label_id, string>& getLabelNames() = 0;

};


class StateHolder: public DataHolder {

public:
	virtual ~StateHolder() {};

	virtual void setSwapListener(SwapListener *listener) = 0;
	virtual void swapSamples(sample_id u, sample_id v) = 0;
	virtual void setCurrentSize(quantity size) = 0;
	virtual quantity getCurrentSize() = 0;
	virtual void reset() = 0;

	virtual sfmatrix* getSamples() = 0;
	virtual label_id* getLabels() = 0;
	virtual map<label_id, string>& getLabelNames() = 0;

	virtual quantity getSize() = 0;
	virtual quantity getSvNumber() = 0;

};

class AbstractSolver: public Solver, public StateHolder {

	SwapListener *listener;

protected:
	TrainParams params;
	StopCriterionStrategy *stopStrategy;

	map<label_id, string> labelNames;

	quantity dimension;
	quantity size;
	quantity currentSize;

	sfmatrix *samples;
	label_id *labels;

	SolverStrategy strategy;

	CachedKernelEvaluator *cache;

protected:
	virtual CachedKernelEvaluator* buildCache(fvalue c, CGaussKernel &gparams);
	void trainForCache(CachedKernelEvaluator *cache);
	void refreshDistr();

public:
	AbstractSolver(map<label_id, string> labelNames, sfmatrix *samples, label_id *labels, TrainParams &params, StopCriterionStrategy *stopStrategy);
	virtual ~AbstractSolver();

	void setKernelParams(fvalue c, CGaussKernel &params);
	virtual void train() = 0;
	PairwiseClassifier getClassifier() = 0;

	void setSwapListener(SwapListener *listener);
	void swapSamples(sample_id u, sample_id v);
	void setCurrentSize(quantity size);
	quantity getCurrentSize();
	void reset();

	sfmatrix* getSamples();
	label_id* getLabels();
	map<label_id, string>& getLabelNames();

	quantity getSize();
	virtual quantity getSvNumber();
};

#endif
