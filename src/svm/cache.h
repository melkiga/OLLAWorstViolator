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
template<typename Strategy>
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
	Strategy *strategy;

	SwapListener *listener;

protected:
	void initialize();
	CacheDimension findCacheDimension(quantity maxSize, quantity problemSize);

	void resizeCache(); // TODO: not sure in which case we would need this

	void evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector *result);

public:
	CachedKernelEvaluator(RbfKernelEvaluator *evaluator, Strategy *strategy, quantity probSize, quantity cchSize, SwapListener *listener);
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

template<typename Strategy>
CachedKernelEvaluator<Strategy>::CachedKernelEvaluator(RbfKernelEvaluator *evaluator, Strategy *strategy, quantity probSize, quantity cchSize, SwapListener *listener) :
		evaluator(evaluator),
		strategy(strategy),
		listener(listener) {
	problemSize = probSize;
	currentSize = problemSize;
	quantity fvaluePerMb = 1024 * 1024 / sizeof(fvalue);
	cacheSize = max(cchSize * fvaluePerMb, 2 * probSize);
	if (cacheSize / probSize > probSize) {
		cacheSize = probSize * probSize;
	}
	CacheDimension dim = findCacheDimension(cacheSize, problemSize);
	cacheDepth = dim.depth;
	cacheLines = dim.lines;
	cache = new fvalue[cacheSize];

	// initialize alphas and output vector
	svnumber = 1;
	alphas = vector<fvalue>(problemSize);
	alphasView = fvectorv_array(alphas.data(), svnumber);
	output = vector<fvalue>(problemSize);
	outputView = fvectorv_array(output.data(), problemSize);

	kernelVector = fvector_alloc(problemSize);

	fbuffer = fvector_alloc(problemSize);
	fbufferView = fvector_subv(fbuffer, 0, svnumber);

	forwardOrder = vector<sample_id>(problemSize);
	backwardOrder = vector<sample_id>(problemSize);
	for (quantity i = 0; i < problemSize; i++) {
		forwardOrder[i] = i;
		backwardOrder[i] = i;
	}

	// initialize cache entries
	views = new fvectorv[cacheLines]; // NOTE: views is a vector of vectors (holds the kernel), cacheLines here is problemSize
	mappings = new EntryMapping[problemSize];
	entries = new CacheEntry[cacheLines];

	initialize();
}

template<typename Strategy>
CachedKernelEvaluator<Strategy>::~CachedKernelEvaluator() {
	delete evaluator;
	delete listener;
	delete [] cache;
	delete [] views;
	delete [] mappings;
	delete [] entries;
	fvector_free(fbuffer);
	fvector_free(kernelVector);
}

template<typename Strategy>
fvalue CachedKernelEvaluator<Strategy>::checkViolation(sample_id v) {
	return output[v]*getLabel(v);
}

/*
* Sets model label to be (+1 or -1) depending on which is the first training pair
*/
template<typename Strategy>
inline void CachedKernelEvaluator<Strategy>::setLabel(pair<label_id, label_id> trainPair) {
	evaluator->setLabel(trainPair.second);
}

/*
 * Returns the label (+1 or -1)
 */
template<typename Strategy>
inline fvalue CachedKernelEvaluator<Strategy>::getLabel(sample_id v) {
	return evaluator->getLabel(v);
}

template<typename Strategy>
void CachedKernelEvaluator<Strategy>::resizeCache() {
	quantity newCacheDepth = min((quantity) (CACHE_DEPTH_INCREASE * cacheDepth), problemSize);
	quantity newCacheLines = min(cacheSize / newCacheDepth, problemSize);
	fvalue *newCache = new fvalue[cacheSize];

	CacheEntry &lastInvalid = entries[lruEntry];

	entry_id entry = lastInvalid.prev;
	for (quantity i = 0; i < newCacheLines; i++) {
		CacheEntry& current = entries[entry];

		fvectorv oldView = views[current.vector];
		size_t oldSize = oldView.vector.size;
		fvectorv newView;
		if (oldSize > 0) {
			newView = fvectorv_array(newCache + i * newCacheDepth, oldView.vector.size);
			fvector_cpy(&newView.vector, &oldView.vector);
		} else {
			newView = fvectorv_array(newCache + i * newCacheDepth, 1);
			newView.vector.size = 0;
		}
		views[current.vector] = newView;
		entry = current.prev;
	}

	if (newCacheLines < cacheLines) {
		CacheEntry &firstInvalid = entries[entry];
		CacheEntry &lastValid = entries[firstInvalid.next];

		for (quantity i = newCacheLines; i < cacheLines; i++) {
			CacheEntry& current = entries[entry];

			EntryMapping &mapping = mappings[current.mapping];
			mapping.cacheEntry = INVALID_ENTRY_ID;

			entry = current.prev;
		}

		CacheEntry &firstValid = entries[lastInvalid.prev];

		lastValid.prev = lastInvalid.prev;
		firstValid.next = firstInvalid.next;
		lruEntry = firstInvalid.next;

		cacheLines = newCacheLines;
	}

	delete [] cache;
	cache = newCache;
	cacheDepth = newCacheDepth;
}

/*
Returns the current number of support vectors.
*/
template<typename Strategy>
inline quantity CachedKernelEvaluator<Strategy>::getSVNumber() {
	return svnumber;
}

/* Returns the worst violator (index and corresponding error), 
	i.e. the sample with the largest error, excluding the current support vectors. 
*/
template<typename Strategy>
CWorstViolator CachedKernelEvaluator<Strategy>::findWorstViolator() {
	fvalue currentWorstError = INT_MAX;
	sample_id currentWorstErrorIndex = INT_MAX;
	fvalue error = 0.0;
	quantity svnumber = getSVNumber();
	CWorstViolator worstViolator(svnumber, currentWorstError);
	for (sample_id i = svnumber; i < currentSize; i++) {
		error = output[i] * getLabel(i);
		if (error < currentWorstError) {
			worstViolator.m_violatorID = i;
			worstViolator.m_error = error;
			currentWorstError = error;
			currentWorstErrorIndex = backwardOrder[i];
		}
		//else if (ksi == min_val && backwardOrder[i] < min_ind) {
		//	worst_viol.violator = i;
		//	worst_viol.yo = ksi;
		//	min_val = ksi;
		//	min_ind = backwardOrder[i];
		//}
	}
	return worstViolator;
}

/*
 * Updates the output vector, worst violator (WV) alpha value, and bias. This is the SGD update step of the L1SVM for OLLAWV. 
 * First, the WV's kernel vector with respect to samples that are non-support vectors is calculated. Next, the kernel vector is multiplied 
 * by the gradient, added to the output vector, as well as the bias update. (output = output + update*K + biasUpdate). Finally, the 
 * WV alpha is updated and the bias too.
 */
template<typename Strategy>
void CachedKernelEvaluator<Strategy>::performSGDUpdate(sample_id worstViolator, fvalue gradient, fvalue biasGradient) {
	// get kernel vector with respect to v
	evalKernel(worstViolator, svnumber, currentSize, kernelVector);

	// update output
	for (int i = svnumber; i < currentSize; i++) {
		output[i] = output[i] + kernelVector->data[i] * gradient + biasGradient;
	}

	// update alphas
	alphas[worstViolator] += gradient;
	updateBias(biasGradient);
}


template<typename Strategy>
void CachedKernelEvaluator<Strategy>::setSwapListener(SwapListener *listener) {
	this->listener = listener;
}

/*
 * Swaps the cache attribute samples when a new support vector is found. The support vectors are
 * stacked at the top of each of the attribute values.
 */
template<typename Strategy>
void CachedKernelEvaluator<Strategy>::swapSamples(sample_id u, sample_id v) {
	evaluator->swapSamples(u, v);
	swap(output[u], output[v]);

	strategy->notifyExchange(u, v);
	if (listener) {
		listener->notify(u, v);
	}

	EntryMapping &vmap = mappings[v];
	EntryMapping &umap = mappings[u];
	entry_id temp = vmap.cacheEntry;
	vmap.cacheEntry = umap.cacheEntry;
	umap.cacheEntry = temp;

	if (umap.cacheEntry != INVALID_ENTRY_ID) {
		entries[umap.cacheEntry].mapping = u;
	}
	if (vmap.cacheEntry != INVALID_ENTRY_ID) {
		entries[vmap.cacheEntry].mapping = v;
	}

	forwardOrder[backwardOrder[u]] = v;
	forwardOrder[backwardOrder[v]] = u;
	swap(backwardOrder[u], backwardOrder[v]);
}

/* 
 * Resets the cache based on the current dimension.
 * Calls initalize when the cache dimension is found.
 */
template<typename Strategy>
void CachedKernelEvaluator<Strategy>::reset() {
	CacheDimension dim = findCacheDimension(cacheSize, problemSize);
	cacheLines = dim.lines;
	cacheDepth = dim.depth;

	initialize();
}

/* 
 * Initialize the cache. Allocates memory for the alphas, output values.
 */
template<typename Strategy>
void CachedKernelEvaluator<Strategy>::initialize() {
	// initialize alphas and kernel values
	for (sample_id i = 0; i < problemSize; i++) {
		alphas[i] = 0.0;
		output[i] = 0.0;
	}

	svnumber = 1;
	alphasView = fvectorv_array(alphas.data(), svnumber);
	outputView = fvectorv_array(output.data(), problemSize);

	// initialize buffer
	fbufferView = fvector_subv(fbuffer, 0, svnumber);

	// initialize vector views
	quantity offset = 0;
	for (quantity i = 0; i < cacheLines; i++) {
		views[i] = fvectorv_array(cache + offset, cacheDepth);
		views[i].vector.size = 0;
		offset += cacheDepth;
	}

	// initialize cache mappings
	fvector *initialVector = &views[INITIAL_ID].vector;
	initialVector->size = 1;
	mappings[INITIAL_ID].cacheEntry = INITIAL_ID;

	// initialize cache entries
	for (entry_id i = INITIAL_ID; i < cacheLines; i++) {
		CacheEntry &entry = entries[i];
		entry.prev = i + 1;
		entry.next = i - 1;
		entry.vector = i;
		entry.mapping = i;

		EntryMapping &mapping = mappings[i];
		mapping.cacheEntry = i;
	}
	for (entry_id i = cacheLines; i < problemSize; i++) {
		EntryMapping &mapping = mappings[i];
		mapping.cacheEntry = INVALID_ENTRY_ID;
	}
	entries[cacheLines - 1].prev = INITIAL_ID;
	entries[INITIAL_ID].next = cacheLines - 1;

	lruEntry = cacheLines - 1;

	evaluator->resetBias();
}

template<typename Strategy>
CacheDimension CachedKernelEvaluator<Strategy>::findCacheDimension(
		quantity cacheSize, quantity problemSize) {
	CacheDimension dimension;
	if (cacheSize / problemSize < problemSize) {
		dimension.depth = max((quantity) INITIAL_CACHE_DEPTH, cacheSize / problemSize);
		dimension.lines = min(cacheSize / dimension.depth, problemSize);
	} else {
		dimension.depth = problemSize;
		dimension.lines = problemSize;
	}
	return dimension;
}

/*
 * Sets the current kernel parameters.
 */
template<typename Strategy>
void CachedKernelEvaluator<Strategy>::setKernelParams(fvalue c, CGaussKernel gparams) {
	evaluator->setKernelParams(c, gparams);
	reset();
}

/*
 * Update the location of the current worst-violator, aka: support vector (SV). Swap the current SV with the first non-SV,
 * update it's location (which is now svnumber). Increment the number of support vectors and the alphas view size.
 */
template<typename Strategy>
void CachedKernelEvaluator<Strategy>::performSvUpdate(sample_id& worstViolator) {
	// TODO: have a look at what's going on here.
	if (svnumber >= cacheDepth) {
		resizeCache();
	}
	else {
		// swap rows
		swapSamples(worstViolator, svnumber);
		worstViolator = svnumber;
	}

	// adjust sv number
	svnumber++;
	alphasView.vector.size++;
}

/*
 * Evaluate the Gaussian RBF Kernel vector with respect to sample 'id' against samples in range 'rangeFrom' to 'rangeTo'.
 * Store the result in 'result'
 */
template<typename Strategy>
inline void CachedKernelEvaluator<Strategy>::evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector *result) {
	evaluator->evalKernel(id, rangeFrom, rangeTo, result);
}

/*
 * Returns the current kernel parameters of the evaluator. 
 */
template<typename Strategy>
inline CGaussKernel CachedKernelEvaluator<Strategy>::getParams() {
	return evaluator->getParams();
}

template<typename Strategy>
inline fvalue CachedKernelEvaluator<Strategy>::getC() {
	return evaluator->getC();
}

template<typename Strategy>
inline fvalue CachedKernelEvaluator<Strategy>::getMargin() {
	return evaluator->getMargin();
}

template<typename Strategy>
inline fvalue CachedKernelEvaluator<Strategy>::getEpochs() {
	return evaluator->getEpochs();
}

template<typename Strategy>
inline fvalue CachedKernelEvaluator<Strategy>::getBetta() {
	return evaluator->getBetta();
}

template<typename Strategy>
inline RbfKernelEvaluator* CachedKernelEvaluator<Strategy>::getEvaluator() {
	return evaluator;
}

template<typename Strategy>
inline vector<fvalue>& CachedKernelEvaluator<Strategy>::getAlphas() {
	return alphas;
}

template<typename Strategy>
inline fvector* CachedKernelEvaluator<Strategy>::getAlphasView() {
	return &alphasView.vector;
}

template<typename Strategy>
inline fvector* CachedKernelEvaluator<Strategy>::getBuffer() {
	return &fbufferView.vector;
}

template<typename Strategy>
vector<sample_id>& CachedKernelEvaluator<Strategy>::getBackwardOrder() {
	return backwardOrder;
}

template<typename Strategy>
vector<sample_id>& CachedKernelEvaluator<Strategy>::getForwardOrder() {
	return forwardOrder;
}

template<typename Strategy>
void CachedKernelEvaluator<Strategy>::updateBias(fvalue LB) {
	evaluator->updateBias(LB);
}

template<typename Strategy>
fvalue CachedKernelEvaluator<Strategy>::getBias() {
	return evaluator->getBias();
}

template<typename Strategy>
void CachedKernelEvaluator<Strategy>::setCurrentSize(quantity size) {
	currentSize = size;
}

#endif
