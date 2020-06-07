#include "svm/cache.h"

CachedKernelEvaluator::CachedKernelEvaluator(RbfKernelEvaluator *evaluator, SolverStrategy *strategy, quantity probSize, quantity cchSize, SwapListener *listener) :
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

CachedKernelEvaluator::~CachedKernelEvaluator() {
	delete evaluator;
	delete listener;
	delete [] cache;
	delete [] views;
	delete [] mappings;
	delete [] entries;
	fvector_free(fbuffer);
	fvector_free(kernelVector);
}

fvalue CachedKernelEvaluator::checkViolation(sample_id v) {
	return output[v]*getLabel(v);
}

void CachedKernelEvaluator::resizeCache() {
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

/* Returns the worst violator (index and corresponding error), 
	i.e. the sample with the largest error, excluding the current support vectors. 
*/
CWorstViolator CachedKernelEvaluator::findWorstViolator() {
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
void CachedKernelEvaluator::performSGDUpdate(sample_id worstViolator, fvalue gradient, fvalue biasGradient) {
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


void CachedKernelEvaluator::setSwapListener(SwapListener *listener) {
	this->listener = listener;
}

/*
 * Swaps the cache attribute samples when a new support vector is found. The support vectors are
 * stacked at the top of each of the attribute values.
 */
void CachedKernelEvaluator::swapSamples(sample_id u, sample_id v) {
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
void CachedKernelEvaluator::reset() {
	CacheDimension dim = findCacheDimension(cacheSize, problemSize);
	cacheLines = dim.lines;
	cacheDepth = dim.depth;

	initialize();
}

/* 
 * Initialize the cache. Allocates memory for the alphas, output values.
 */
void CachedKernelEvaluator::initialize() {
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

CacheDimension CachedKernelEvaluator::findCacheDimension(
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
void CachedKernelEvaluator::setKernelParams(fvalue c, CGaussKernel gparams) {
	evaluator->setKernelParams(c, gparams);
	reset();
}

/*
 * Update the location of the current worst-violator, aka: support vector (SV). Swap the current SV with the first non-SV,
 * update it's location (which is now svnumber). Increment the number of support vectors and the alphas view size.
 */
void CachedKernelEvaluator::performSvUpdate(sample_id& worstViolator) {
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

vector<sample_id>& CachedKernelEvaluator::getBackwardOrder() {
	return backwardOrder;
}


vector<sample_id>& CachedKernelEvaluator::getForwardOrder() {
	return forwardOrder;
}


void CachedKernelEvaluator::updateBias(fvalue LB) {
	evaluator->updateBias(LB);
}


fvalue CachedKernelEvaluator::getBias() {
	return evaluator->getBias();
}


void CachedKernelEvaluator::setCurrentSize(quantity size) {
	currentSize = size;
}