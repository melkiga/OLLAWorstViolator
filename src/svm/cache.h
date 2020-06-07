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

#ifndef CACHE_H_
#define CACHE_H_

#include <set>

#include "svm/strategy.h"
#include "kernel.h"
#include "../math/random.h"

// uncomment to enable statistics
//#define ENABLE_STATS

#ifdef ENABLE_STATS
#define STATS(c) c
#else
#define STATS(c)
#endif

#define CACHE_DENSITY_RATIO 0.1
#define CACHE_DEPTH_INCREASE 1.5
#define INITIAL_CACHE_DEPTH 256
#define INITIAL_ID 0

typedef sample_id row_id;

typedef short_id fold_id;

typedef unsigned long iteration;


struct EntryMapping {

	entry_id cacheEntry;

	EntryMapping() :
		cacheEntry(INVALID_ENTRY_ID) {
	}

};

/*
 * Worst violating vector structure that holds the index of the current worst violator and it's error.
 */
struct CWorstViolator {

	sample_id m_violatorID;
	fvalue m_error;

	CWorstViolator(sample_id violatorID, fvalue error) :
		m_violatorID(violatorID),
		m_error(error) {
	}

};

struct CacheEntry {

	entry_id prev;
	entry_id next;

	entry_id vector;
	sample_id mapping;

	CacheEntry() :
		prev(INVALID_ENTRY_ID),
		next(INVALID_ENTRY_ID),
		vector(INVALID_ENTRY_ID),
		mapping(INVALID_SAMPLE_ID) {
	}

};

class SwapListener {

public:
	virtual ~SwapListener() {};

	virtual void notify(sample_id u, sample_id v) = 0;

};


struct CacheDimension {

	quantity lines;
	quantity depth;

};

/*
Holds the current model. 
*/
class CachedKernelEvaluator {

	vector<fvalue> output;
	fvectorv outputView;

	vector<fvalue> alphas;
	fvectorv alphasView;
	quantity svnumber;

	fvector *kernelVector;

	fvector *fbuffer;
	fvectorv fbufferView;

	quantity problemSize;
	quantity currentSize;

	quantity cacheSize;
	quantity cacheLines;
	quantity cacheDepth;
	fvalue *cache;

	fvectorv *views;
	vector<sample_id> forwardOrder;
	vector<sample_id> backwardOrder;

	EntryMapping *mappings;
	CacheEntry *entries;

	entry_id lruEntry;

	RbfKernelEvaluator *evaluator;
	SolverStrategy *strategy;

	SwapListener *listener;

protected:
	void initialize();
	CacheDimension findCacheDimension(quantity maxSize, quantity problemSize);

	void resizeCache();

	void evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector *result);

public:
	CachedKernelEvaluator(RbfKernelEvaluator *evaluator, SolverStrategy *strategy, quantity probSize, quantity cchSize, SwapListener *listener);
	~CachedKernelEvaluator();

	fvalue checkViolation(sample_id v);
	CWorstViolator findWorstViolator();
	
	fvalue getLabel(sample_id v);
	void setLabel(pair<label_id, label_id> trainPair);
	void setCurrentSize(quantity size);
	quantity getSVNumber();

	void performSGDUpdate(sample_id worstViolator, fvalue gradient, fvalue biasGradient);
	void performSvUpdate(sample_id& v);

	void setSwapListener(SwapListener *listener);
	void swapSamples(sample_id u, sample_id v);
	void reset();
	void setKernelParams(fvalue c, CGaussKernel params);

  CGaussKernel getParams();
	fvalue getC();
	RbfKernelEvaluator* getEvaluator();
	vector<fvalue>& getAlphas();
	fvector* getAlphasView();
	fvector* getBuffer();
	vector<sample_id>& getBackwardOrder();
	vector<sample_id>& getForwardOrder();

	void updateBias(fvalue LB);
	fvalue getBias();
	fvalue getBetta();
	fvalue getEpochs();
	fvalue getMargin();
};

/*
* Sets model label to be (+1 or -1) depending on which is the first training pair
*/
inline void CachedKernelEvaluator::setLabel(pair<label_id, label_id> trainPair) {
	evaluator->setLabel(trainPair.second);
}

/*
 * Returns the label (+1 or -1)
 */
inline fvalue CachedKernelEvaluator::getLabel(sample_id v) {
	return evaluator->getLabel(v);
}

/*
Returns the current number of support vectors.
*/
inline quantity CachedKernelEvaluator::getSVNumber() {
	return svnumber;
}

/*
 * Evaluate the Gaussian RBF Kernel vector with respect to sample 'id' against samples in range 'rangeFrom' to 'rangeTo'.
 * Store the result in 'result'
 */
inline void CachedKernelEvaluator::evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector *result) {
	evaluator->evalKernel(id, rangeFrom, rangeTo, result);
}

/*
 * Returns the current kernel parameters of the evaluator. 
 */
inline CGaussKernel CachedKernelEvaluator::getParams() {
	return evaluator->getParams();
}


inline fvalue CachedKernelEvaluator::getC() {
	return evaluator->getC();
}


inline fvalue CachedKernelEvaluator::getMargin() {
	return evaluator->getMargin();
}


inline fvalue CachedKernelEvaluator::getEpochs() {
	return evaluator->getEpochs();
}


inline fvalue CachedKernelEvaluator::getBetta() {
	return evaluator->getBetta();
}


inline RbfKernelEvaluator* CachedKernelEvaluator::getEvaluator() {
	return evaluator;
}


inline vector<fvalue>& CachedKernelEvaluator::getAlphas() {
	return alphas;
}


inline fvector* CachedKernelEvaluator::getAlphasView() {
	return &alphasView.vector;
}


inline fvector* CachedKernelEvaluator::getBuffer() {
	return &fbufferView.vector;
}


#endif
