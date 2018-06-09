
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
