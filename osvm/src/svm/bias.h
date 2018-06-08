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

#ifndef BIAS_H_
#define BIAS_H_

#include "kernel.h"
#include "params.h"
#include "../math/numeric.h"

#define DEFAULT_EPSILON 0.1

#define EXPECTED_SV_NUMBER 15000

//TODO: get rid of this - we evaluate bias within the training (no need for a 'strategy').
template<typename Matrix>
class BiasEvaluationStrategy {

public:
	virtual vector<fvalue> getBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c) = 0;
	virtual fvalue getBinaryBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity sampleNumber, label_id label, fvalue rho, fvalue c) = 0;

	virtual ~BiasEvaluationStrategy();

};

template<typename Matrix>
BiasEvaluationStrategy<Matrix>::~BiasEvaluationStrategy() {
}


/**
 * Bias evaluation strategy that uses theoretic dual space formula for bias
 * calculation.
 */
template<typename Matrix>
class TheoreticBiasStrategy: public BiasEvaluationStrategy<Matrix> {

public:
	virtual vector<fvalue> getBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c);
	virtual fvalue getBinaryBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity sampleNumber, label_id label, fvalue rho, fvalue c);

	virtual ~TheoreticBiasStrategy();

};

template<typename Matrix>
TheoreticBiasStrategy<Matrix>::~TheoreticBiasStrategy() {
}

template<typename Matrix>
vector<fvalue> TheoreticBiasStrategy<Matrix>::getBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c) {
	vector<fvalue> bias(labelNumber, YY_NEG(labelNumber));

	fvalue yy = -YY_NEG(labelNumber) + YY_POS;
	for (sample_id v = 0; v < sampleNumber; v++) {
		label_id label = labels[v];
		bias[label] += yy * alphas[v];
	}
	return bias;
}

template<typename Matrix>
fvalue TheoreticBiasStrategy<Matrix>::getBinaryBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity sampleNumber, label_id label, fvalue rho, fvalue c) {
	fvalue bias = 0;
	fvalue mult[] = {YY_NEG(2), YY_POS};
	for (sample_id v = 0; v < sampleNumber; v++) {
		bias += alphas[v] * mult[labels[v] == label];
	}
	return bias;
}


/**
 * Bias evaluation strategy that based on KKT conditions calculates the value
 * of the bias for every support vector and then averages the results.
 */
template<typename Matrix>
class AverageBiasStrategy: public BiasEvaluationStrategy<Matrix> {

	RbfKernelEvaluator<Matrix> *evaluator;

public:
	AverageBiasStrategy(RbfKernelEvaluator<Matrix> *evaluator);

	virtual vector<fvalue> getBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c);
	virtual fvalue getBinaryBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity sampleNumber, label_id label, fvalue rho, fvalue c);

	virtual ~AverageBiasStrategy();

};

template<typename Matrix>
AverageBiasStrategy<Matrix>::AverageBiasStrategy(
		RbfKernelEvaluator<Matrix>* evaluator) :
		evaluator(evaluator) {
}

template<typename Matrix>
AverageBiasStrategy<Matrix>::~AverageBiasStrategy() {
}

template<typename Matrix>
vector<fvalue> AverageBiasStrategy<Matrix>::getBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c) {
	vector<fvalue> bias(labelNumber, 0.0);
	vector<quantity> count(labelNumber, 0);

	fvector* kernelBuffer = fvector_alloc(sampleNumber);

	for (id v = 0; v < sampleNumber; v++) {
		if (alphas[v] > 0.0) {
			label_id label = labels[v];
			fvalue value = rho - alphas[v] / c;

			// TODO use cache instead
			evaluator->evalInnerKernel(v, 0, sampleNumber, kernelBuffer);
			fvalue* kernelValues = kernelBuffer->data;
			for (id u = 0; u < sampleNumber; u++) {
				fvalue yy = (labels[u] == labels[v]) ? YY_POS : YY_NEG(labelNumber);
				value -= yy * alphas[u] * kernelValues[u];
			}
			bias[label] += value;
			count[label]++;
		}
	}
	for (label_id l = 0; l < labelNumber; l++) {
		bias[l] /= count[l];
	}

	fvector_free(kernelBuffer);
	return bias;
}

template<typename Matrix>
fvalue AverageBiasStrategy<Matrix>::getBinaryBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity sampleNumber, label_id label, fvalue rho, fvalue c) {
	fvalue bias = 0.0;
	quantity count = 0;

	fvalue mult[] = {YY_NEG(2), YY_POS};
	fvector* kernelBuffer = fvector_alloc(sampleNumber);

	for (id v = 0; v < sampleNumber; v++) {
		if (alphas[v] > 0.0) {
			fvalue value = rho - alphas[v] / c;

			// TODO use cache instead
			evaluator->evalInnerKernel(v, 0, sampleNumber, kernelBuffer);
			fvalue* kernelValues = kernelBuffer->data;
			for (id u = 0; u < sampleNumber; u++) {
				fvalue yyuv = mult[labels[v] == labels[u]];
				value -= yyuv * alphas[u] * kernelValues[u];
			}
			fvalue yy = mult[label == labels[v]];
			bias += yy * value;
			count++;
		}
	}

	fvector_free(kernelBuffer);
	return bias / count;
}


/**
 * Bias evaluation strategy that always returns 0. Used in SVM training
 * without bias.
 */
template<typename Matrix>
class NoBiasStrategy: public BiasEvaluationStrategy<Matrix> {

public:
	virtual vector<fvalue> getBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c);
	virtual fvalue getBinaryBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity sampleNumber, label_id label, fvalue rho, fvalue c);

	virtual ~NoBiasStrategy();

};

template<typename Matrix>
NoBiasStrategy<Matrix>::~NoBiasStrategy() {
}

template<typename Matrix>
vector<fvalue> NoBiasStrategy<Matrix>::getBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c) {
	return vector<fvalue>(labelNumber, 0.0);
}

template<typename Matrix>
fvalue NoBiasStrategy<Matrix>::getBinaryBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity sampleNumber, label_id label, fvalue rho, fvalue c) {
	return 0.0;
}


/**
 * Factory class for bias evaluators.
 */
template<typename Matrix>
class BiasEvaluatorFactory {

public:
	BiasEvaluationStrategy<Matrix>* createEvaluator(BiasType type,
			RbfKernelEvaluator<Matrix>* evaluator);

};

template<typename Matrix>
BiasEvaluationStrategy<Matrix>* BiasEvaluatorFactory<Matrix>::createEvaluator(
		BiasType type, RbfKernelEvaluator<Matrix>* evaluator) {
	BiasEvaluationStrategy<Matrix>* eval = NULL;
	if (type == YES) {
		eval = new TheoreticBiasStrategy<Matrix>();
	} else {
		eval = new NoBiasStrategy<Matrix>();
	}
	return eval;
}

#endif
