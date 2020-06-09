
#include "pairwise_solver.h"

PairwiseClassifier::PairwiseClassifier(
  RbfKernelEvaluator *evaluator,
  PairwiseTrainingResult* state, fvector *buffer) :
  evaluator(evaluator),
  state(state),
  buffer(buffer),
  votes(state->totalLabelCount),
  evidence(state->totalLabelCount) {
}

PairwiseClassifier::~PairwiseClassifier() {
}

label_id PairwiseClassifier::classify(sample_id sample) {
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
      maxEvidence = (quantity)evidence[i];
    }
  }

  return maxLabelId;
}

fvalue PairwiseClassifier::getDecisionForModel(sample_id sample,
  PairwiseTrainingModel* model, fvector* buffer) {
  fvalue dec = model->bias;
  fvalue* kernels = buffer->data;
  for (sample_id i = 0; i < model->size; i++) {
    dec += model->yalphas[i] * kernels[model->samples[i]];
  }
  return dec;
}

inline fvalue PairwiseClassifier::convertDecisionToEvidence(
  fvalue decision) {
  return decision;
}

quantity PairwiseClassifier::getSvNumber() {
  return state->maxSVCount;
}

void PairwiseClassifier::saveClassifier(string testName){
	// initialize json property tree
	pt::ptree root;

	// get pairwise models
	quantity maxSVCount = state->maxSVCount;
	root.put("maxSVCount",maxSVCount);
	
	// get pairwise models
	pt::ptree models;
	int counter = 0;
	vector<PairwiseTrainingModel>::iterator it;
	for (it = state->models.begin(); it != state->models.end(); it++) {
		pt::ptree state;
		state.put("bias",it->bias);
		state.put("size",it->size);
		state.put("labels", "[" + to_string(it->trainingLabels.first) + ", " + to_string(it->trainingLabels.second) + "]");

		vector<sample_id> samples(it->samples.begin(),it->samples.begin()+maxSVCount);
		vector<fvalue> alphas(it->yalphas.begin(),it->yalphas.begin()+maxSVCount);
		string alphalist = "[", samplelist = "[";
		for(auto const& i: boost::combine(alphas,samples)){
			fvalue alpha;
			sample_id sample;
			boost::tie(alpha,sample) = i;
			alphalist += to_string(alpha) + ", ";
			samplelist += to_string(sample) + ", ";
		}
		samplelist.pop_back(), samplelist.pop_back();
		alphalist.pop_back(), alphalist.pop_back();
		alphalist += "]", samplelist += "]";
		state.put("alphas",alphalist), state.put("samples",samplelist);
		// save sub-model
		models.push_back(make_pair(to_string(counter),state));
		counter++;
	}

	root.add_child("models",models);
	//pt::write_json(std::cout, root);
	pt::write_json(testName, root);	
}

PairwiseSolver::PairwiseSolver(
		map<label_id, string> labelNames, sfmatrix *samples,
		label_id *labels, TrainParams &params,
		StopCriterionStrategy *stopStrategy) :
		AbstractSolver(labelNames,
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


PairwiseSolver::~PairwiseSolver() {
}


Classifier* PairwiseSolver::getClassifier() {
	return new PairwiseClassifier(this->cache->getEvaluator(), &state, this->cache->getBuffer());
}


void PairwiseSolver::train() {
	RbfKernelEvaluator* evaluator = this->cache->getEvaluator();

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


quantity PairwiseSolver::reorderSamples(label_id *labels, quantity size, pair<label_id, label_id>& labelPair) {
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


CachedKernelEvaluator* PairwiseSolver::buildCache(fvalue c, CGaussKernel &gparams) {
	fvalue bias = (this->params.bias == NO) ? 0.0 : 1.0;
	RbfKernelEvaluator *rbf = new RbfKernelEvaluator(this->samples, this->labels, 2, bias, c, gparams, this->params.epochs, this->params.margin);
	return new CachedKernelEvaluator(rbf, &this->strategy, this->size, this->params.cache.size, NULL);
}


quantity PairwiseSolver::getSvNumber() {
	return state.maxSVCount;
}