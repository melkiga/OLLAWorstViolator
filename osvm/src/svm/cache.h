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


struct Stats {

	struct KernelStats {
		quantity kernelEvaluations;
	} kernel;

	Stats() {
		kernel.kernelEvaluations = 0;
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
template<typename Kernel, typename Matrix, typename Strategy>
class CachedKernelEvaluator {

	vector<fvalue> output;
	fvectorv outputView;

	vector<fvalue> alphas;
	fvectorv alphasView;
	quantity svnumber;
	quantity psvnumber;

	fvector *kernelVector;

	fvector *fbuffer;
	fvectorv fbufferView;

	quantity problemSize;
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

	fvalue w2;

	RbfKernelEvaluator<Kernel, Matrix> *evaluator;
	Strategy *strategy;

	SwapListener *listener;

	Stats stats;

protected:
	void initialize();
	CacheDimension findCacheDimension(quantity maxSize, quantity problemSize);

	CacheEntry& initializeEntry(sample_id v);
	void refreshEntry(sample_id v);

	fvector& evalKernelVector(sample_id v);

	void resizeCache();

	fvalue evalKernel(sample_id uid, sample_id vid);
	void evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector *result);

public:
	CachedKernelEvaluator(RbfKernelEvaluator<Kernel, Matrix> *evaluator, Strategy *strategy, quantity probSize, quantity cchSize, SwapListener *listener);
	~CachedKernelEvaluator();

	fvalue evalKernelUV(sample_id u, sample_id v);
	fvalue evalKernelAXV(sample_id v);
	fvalue checkViolation(sample_id v);
	
	fvalue getLabel(sample_id v);
	void setLabel(pair<label_id, label_id> trainPair);

	fvalue getVectorWeight(sample_id v);
	quantity getSVNumber();

	void performUpdate(sample_id v, fvalue lambda, fvalue LB);
	fvalue getWNorm();
	vector<fvalue>& getKernelValues();
	void performSvUpdate(sample_id& v);

	void setSwapListener(SwapListener *listener);
	void swapSamples(sample_id u, sample_id v);
	void reset();
	void shrink();
	void releaseSupportVectors(fold_id *membership, fold_id fold);
	void setKernelParams(fvalue c, Kernel params);

	Kernel getParams();
	fvalue getC();
	RbfKernelEvaluator<Kernel, Matrix>* getEvaluator();
	vector<fvalue>& getAlphas();
	fvector* getAlphasView();
	fvector* getBuffer();
	vector<sample_id>& getBackwardOrder();
	vector<sample_id>& getForwardOrder();

	void updateBias(fvalue LB);
	fvalue getBias();

	void structureCheck();
	void reportStatistics();
};

template<typename Kernel, typename Matrix, typename Strategy>
CachedKernelEvaluator<Kernel, Matrix, Strategy>::CachedKernelEvaluator(
		RbfKernelEvaluator<Kernel, Matrix> *evaluator, Strategy *strategy,
		quantity probSize, quantity cchSize, SwapListener *listener) :
		evaluator(evaluator),
		strategy(strategy),
		listener(listener) {
	problemSize = probSize;
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

template<typename Kernel, typename Matrix, typename Strategy>
CachedKernelEvaluator<Kernel, Matrix, Strategy>::~CachedKernelEvaluator() {
	delete evaluator;
	delete listener;
	delete [] cache;
	delete [] views;
	delete [] mappings;
	delete [] entries;
	fvector_free(fbuffer);
	fvector_free(kernelVector);
}

template<typename Kernel, typename Matrix, typename Strategy>
fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernelAXV(sample_id v) {
	fvalue result;
	if (v >= svnumber) {
		fvector &vector = evalKernelVector(v);
		fvector_sum(&outputView.vector);
		// TODO: check what's going on here.

		fvector_dot(&alphasView.vector, &vector, &result);
		// save current result (valid until alpha values stay unchanged)
		//kernelValues[v] = result;
	} else {
		// get precomputed kernel value
		//result = kernelValues[v];
		refreshEntry(v);
	}
	return result;
}

template<typename Kernel, typename Matrix, typename Strategy>
fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::checkViolation(sample_id v) {
	return output[v]*getLabel(v);
}

/*
* Sets model label to be (+1 or -1) depending on which is the first training pair
*/
template<typename Kernel, typename Matrix, typename Strategy>
inline void CachedKernelEvaluator<Kernel, Matrix, Strategy>::setLabel(pair<label_id, label_id> trainPair) {
	evaluator->setLabel(trainPair.second);
}

/*
 * Returns the label (+1 or -1)
 */
template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::getLabel(sample_id v) {
	return evaluator->getLabel(v);
}

template<typename Kernel, typename Matrix, typename Strategy>
fvector& CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernelVector(sample_id v) {
	evalKernel(v, 0, problemSize, kernelVector);
	return *kernelVector;
}

template<typename Kernel, typename Matrix, typename Strategy>
CacheEntry& CachedKernelEvaluator<Kernel, Matrix, Strategy>::initializeEntry(sample_id v) {
	// entry to be initialized
	CacheEntry &released = entries[lruEntry];

	// clear previous mapping
	EntryMapping &prevMapp = mappings[released.mapping];
	prevMapp.cacheEntry = INVALID_ENTRY_ID;
	released.mapping = v;

	// initialize current mapping
	EntryMapping &currMapp = mappings[v];
	currMapp.cacheEntry = lruEntry;

	// scroll LRU pointer
	lruEntry = released.next;
	return released;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::refreshEntry(sample_id v) {
	CacheEntry &lru = entries[lruEntry];

	entry_id currentId = mappings[v].cacheEntry;
	if (currentId != INVALID_ENTRY_ID) {
		if (currentId != lruEntry) {
			CacheEntry& current = entries[currentId];

			CacheEntry &prev = entries[current.prev];
			prev.next = current.next;

			CacheEntry &next = entries[current.next];
			next.prev = current.prev;

			CacheEntry &lruPrev = entries[lru.prev];

			current.next = lruEntry;
			current.prev = lru.prev;

			lru.prev = currentId;
			lruPrev.next = currentId;
		} else {
			lruEntry = lru.next;
		}
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::resizeCache() {
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

	//	shrink();
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::shrink() {
	// removing non-support vectors
	quantity *nonsvnumbers = new quantity[svnumber + 1];
	quantity nonsvnumber = 0;
	sample_id firstViol = 0;
	nonsvnumbers[0] = 0;
	for (quantity i = 1; i <= svnumber; i++) {
		if (alphas[i - 1] <= 0) {
			nonsvnumbers[i] = nonsvnumbers[i - 1] + 1;
			nonsvnumber++;
			if (!firstViol) {
				firstViol = i;
			}
		} else {
			nonsvnumbers[i] = nonsvnumbers[i - 1];
		}
	}
	if (nonsvnumber > 0) {
		// update vector lengths
		for (sample_id i = 0; i < svnumber; i++) {
			if (mappings[i].cacheEntry != INVALID_ENTRY_ID) {
				CacheEntry &entry = entries[mappings[i].cacheEntry];
				fvectorv &vector = views[entry.vector];
				fvalue *data = vector.vector.data;
				for (quantity j = firstViol; j <= vector.vector.size; j++) {
					data[j - nonsvnumbers[j]] = data[j];
				}
				vector.vector.size -= nonsvnumbers[vector.vector.size];
			}
			if (nonsvnumbers[i] > 0) {
				sample_id svid = i - nonsvnumbers[i];
				swapSamples(i, svid);
			}
		}

		for (sample_id i = svnumber; i < problemSize; i++) {
			if (mappings[i].cacheEntry != INVALID_ENTRY_ID) {
				CacheEntry &entry = entries[mappings[i].cacheEntry];
				fvectorv &vector = views[entry.vector];
				vector.vector.size = 0;
			}
		}

		svnumber -= nonsvnumber;
		psvnumber = svnumber;
		alphasView.vector.size -= nonsvnumber;
		outputView.vector.size -= nonsvnumber;
	}
	delete [] nonsvnumbers;
}

template<typename Kernel, typename Matrix, typename Strategy>
fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernelUV(sample_id u, sample_id v) {
	fvalue result;
	if (mappings[v].cacheEntry != INVALID_ENTRY_ID
			&& views[mappings[v].cacheEntry].vector.size > u) {
		result = fvector_get(&views[mappings[v].cacheEntry].vector, u);
	} else if (mappings[u].cacheEntry != INVALID_ENTRY_ID
			&& views[mappings[u].cacheEntry].vector.size > v) {
		result = fvector_get(&views[mappings[u].cacheEntry].vector, v);
	} else {
		// calculate the kernel, do not cache the value
		result = evalKernel(u, v);
	}
	return result;
}

/*
Returns alpha value of input sample id.
*/
template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::getVectorWeight(sample_id v) {
	return alphas[v];
}

/*
Returns the current number of support vectors.
*/
template<typename Kernel, typename Matrix, typename Strategy>
inline quantity CachedKernelEvaluator<Kernel, Matrix, Strategy>::getSVNumber() {
	return svnumber;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::performUpdate(sample_id v, fvalue lambda, fvalue LB) {
	// update output
	fvector &vector = evalKernelVector(v);
	fvector_mul_const(&vector, lambda);
	fvector_add(&outputView.vector, &vector);
	fvector_add_const(&outputView.vector, LB);
	// update alphas
	alphas[v] += lambda;
	updateBias(LB);
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::releaseSupportVectors(fold_id *membership, fold_id fold) {
	quantity valid = 0;
	for (sample_id v = 0; v < svnumber; v++) {
		fvalue beta = alphas[v];
		if (membership[v] != fold && beta > 0) {
			valid++;
		}
	}
	if (valid > 0) {
		for (sample_id v = 0; v < svnumber; v++) {
			fvalue beta = alphas[v];
			if (membership[v] == fold && beta > 0) {
				// update alphas
				alphas[v] = 0.0;
			}
		}
	} else {
		sample_id proper = svnumber;
		while (membership[proper] == fold) {
			proper++;
		}
		swapSamples(0, proper);
		reset();
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::setSwapListener(SwapListener *listener) {
	this->listener = listener;
}

/*
 * Swaps the cache attribute samples when a new support vector is found. The support vectors are
 * stacked at the top of each of the attribute values.
 */
template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::swapSamples(sample_id u, sample_id v) {
	evaluator->swapSamples(u, v);
	swap(alphas[u], alphas[v]);
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
template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::reset() {
	CacheDimension dim = findCacheDimension(cacheSize, problemSize);
	cacheLines = dim.lines;
	cacheDepth = dim.depth;

	initialize();
}

/* 
 * Initialize the cache. Allocates memory for the alphas, output values.
 */
template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::initialize() {
	// initialize alphas and kernel values
	for (sample_id i = 0; i < problemSize; i++) {
		alphas[i] = 0.0;
		output[i] = 0.0;
	}

	svnumber = 1;
	psvnumber = 1;
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
	initialVector->size = 1; //TODO: initialize kernel vector here, so we don't have to in the train()
	//fvector_set(initialVector, 0, evaluator->getKernelTau());
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

	w2 = evaluator->getKernelTau();
	evaluator->resetBias();
}

template<typename Kernel, typename Matrix, typename Strategy>
CacheDimension CachedKernelEvaluator<Kernel, Matrix, Strategy>::findCacheDimension(
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
template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::setKernelParams(fvalue c, Kernel gparams) {
	evaluator->setKernelParams(c, gparams);
	reset();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::getWNorm() {
	return w2;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline vector<fvalue>& CachedKernelEvaluator<Kernel, Matrix, Strategy>::getKernelValues() {
	return kernelValues;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::performSvUpdate(sample_id& v) {
	
	if (v >= svnumber) {
		if (svnumber >= cacheDepth) {
			resizeCache();
		}

		// swap rows
		swapSamples(v, svnumber);
		v = svnumber;

		// adjust sv number
		svnumber++;
		alphasView.vector.size++;
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernel(sample_id uid, sample_id vid) {
	STATS({
		stats.kernel.kernelEvaluations++;
	});
	return evaluator->evalKernel(uid, vid);
}

template<typename Kernel, typename Matrix, typename Strategy>
inline void CachedKernelEvaluator<Kernel, Matrix, Strategy>::evalKernel(sample_id id,
		sample_id rangeFrom, sample_id rangeTo, fvector *result) {
	STATS({
		stats.kernel.kernelEvaluations += rangeTo - rangeFrom;
	});
	evaluator->evalKernel(id, rangeFrom, rangeTo, result);
}

/*
 * Returns the current kernel parameters of the evaluator. 
 */
template<typename Kernel, typename Matrix, typename Strategy>
inline Kernel CachedKernelEvaluator<Kernel, Matrix, Strategy>::getParams() {
	return evaluator->getParams();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::getC() {
	return evaluator->getC();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline RbfKernelEvaluator<Kernel, Matrix>* CachedKernelEvaluator<Kernel, Matrix, Strategy>::getEvaluator() {
	return evaluator;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline vector<fvalue>& CachedKernelEvaluator<Kernel, Matrix, Strategy>::getAlphas() {
	return alphas;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvector* CachedKernelEvaluator<Kernel, Matrix, Strategy>::getAlphasView() {
	return &alphasView.vector;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline fvector* CachedKernelEvaluator<Kernel, Matrix, Strategy>::getBuffer() {
	return &fbufferView.vector;
}

template<typename Kernel, typename Matrix, typename Strategy>
vector<sample_id>& CachedKernelEvaluator<Kernel, Matrix, Strategy>::getBackwardOrder() {
	return backwardOrder;
}

template<typename Kernel, typename Matrix, typename Strategy>
vector<sample_id>& CachedKernelEvaluator<Kernel, Matrix, Strategy>::getForwardOrder() {
	return forwardOrder;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::reportStatistics() {
	logger << format("stats: kernel evaluations: %d\n") % stats.kernel.kernelEvaluations;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::structureCheck() {
	try {
		set<id> ids;
		entry_id ptr = lruEntry;
		for (quantity i = 0; i < cacheLines; i++) {
			ids.insert(ptr);
			ptr = entries[ptr].next;
		}
		if (ptr != lruEntry || ids.size() != cacheLines) {
			throw runtime_error("invalid cache structure (next)");
		}

		ids.clear();

		for (quantity i = 0; i < cacheLines; i++) {
			ids.insert(ptr);
			ptr = entries[ptr].prev;
		}
		if (ptr != lruEntry || ids.size() != cacheLines) {
			throw runtime_error("invalid cache structure (prev)");
		}

		for (sample_id i = 0; i < problemSize; i++) {
			entry_id id = mappings[i].cacheEntry;
			if (id != INVALID_ENTRY_ID && ids.count(id) == 0) {
				throw runtime_error("invalid mapping");
			}

			if (id != INVALID_ENTRY_ID && entries[id].mapping != i) {
				throw runtime_error("invalid entry mapping relation");
			}
		}

		if (fabs(fvector_sum(&alphasView.vector) - 1.0) > 0.0001) {
			throw runtime_error("invalid alphas");
		}

		entry_id entryId = lruEntry;
		for (entry_id e = 0; e < cacheLines; e++) {
			CacheEntry &entry = entries[entryId];
			fvectorv vect = views[entry.vector];
			for (size_t i = 0; i < vect.vector.size; i++) {
				fvalue kern = evalKernel(entry.mapping, i);
				fvalue buff = fvector_get(&vect.vector, i);
				if (fabs(kern - buff) > 0.0001) {
					throw runtime_error("invalid kernel value");
				}
			}
			if (vect.vector.size == svnumber && entry.mapping < svnumber) {
				fvalue kern;
				fvector_dot(&vect.vector, &alphasView.vector, &kern);
				fvalue buff = kernelValues[entry.mapping];
				if (fabs(kern - buff) > 0.0001) {
					throw runtime_error("invalid buffered kernel value");
				}
			}
			entryId = entry.next;
		}

		fvalue w2ex;
		fvector_dot(&kernelValuesView.vector, &alphasView.vector, &w2ex);
		if (fabs(w2ex - w2) > 0.0001) {
			throw runtime_error("invalid w2");
		}
	} catch (runtime_error const& ex) {
		cerr << ex.what() << endl;
		throw ex;
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void CachedKernelEvaluator<Kernel, Matrix, Strategy>::updateBias(fvalue LB) {
	evaluator->updateBias(LB);
}

template<typename Kernel, typename Matrix, typename Strategy>
fvalue CachedKernelEvaluator<Kernel, Matrix, Strategy>::getBias() {
	return evaluator->getBias();
}

#endif
