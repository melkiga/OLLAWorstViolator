#include "solver.h"

AbstractSolver::AbstractSolver(map<label_id, string> labelNames, sfmatrix *samples, label_id *labels, TrainParams &params, StopCriterionStrategy *stopStrategy) :
		params(params),
		stopStrategy(stopStrategy),
		labelNames(labelNames),
		samples(samples),
		labels(labels),
		strategy(SolverStrategy(params, (quantity) labelNames.size(), labels, (quantity) samples->height)) {
	size = (quantity) samples->height;
	currentSize = (quantity) samples->height;
	dimension = (quantity) samples->width;

	listener = NULL;
	cache = NULL;

	refreshDistr();
}

AbstractSolver::~AbstractSolver() {
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

void AbstractSolver::setKernelParams(fvalue c, CGaussKernel &gparams) {
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

void AbstractSolver::trainForCache(CachedKernelEvaluator *cache) 
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


CachedKernelEvaluator* AbstractSolver::buildCache(fvalue c, CGaussKernel &gparams) {
	fvalue bias = (params.bias == NO) ? 0.0 : 1.0;
	RbfKernelEvaluator *rbf = new RbfKernelEvaluator(this->samples, this->labels, (quantity) labelNames.size(), bias, c, gparams, params.epochs, params.margin);
	return new CachedKernelEvaluator(rbf, &strategy, size, params.cache.size, NULL);
}


sfmatrix* AbstractSolver::getSamples() {
	return samples;
}


label_id* AbstractSolver::getLabels() {
	return labels;
}


map<label_id, string>& AbstractSolver::getLabelNames() {
	return labelNames;
}


void AbstractSolver::refreshDistr() {
	this->strategy.resetGenerator(labels, currentSize);
}


void AbstractSolver::setSwapListener(SwapListener *listener) {
	this->listener = listener;
	if (cache) {
		cache->setSwapListener(listener);
	}
}


void AbstractSolver::swapSamples(sample_id u, sample_id v) {
	cache->swapSamples(u, v);
}


void AbstractSolver::reset() {
	cache->reset();
}


void AbstractSolver::setCurrentSize(quantity size) {
	currentSize = size;
	cache->setCurrentSize(size);
	refreshDistr();
}


inline quantity AbstractSolver::getCurrentSize() {
	return currentSize;
}


quantity AbstractSolver::getSize() {
	return size;
}


quantity AbstractSolver::getSvNumber() {
	return this->cache->getSVNumber();
}