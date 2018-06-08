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

#ifndef KERNEL_H_
#define KERNEL_H_

#include "../math/numeric.h"
#include "../math/matrix.h"

#define YY_POS 1.0
#define YY_NEG(cl) (-1.0 / (cl - 1))

/*
 * Gaussian RBF Kernel. negativeGamma is the parameter associated with the Gaussian curves broadness. 
 * Gaussian kernel matrix = e^(-gamma * ||X - C||^2) where ||o|| is the euclidean distance
 */
struct CGaussKernel {
	fvalue m_negativeGamma;

	CGaussKernel(fvalue gamma);

	fvalue m_evaluateKernel(fvalue euclideanDistanceSquared);

};

/*
 * Evaluate the gaussian kernel given a euclidean distance squared. (||x - c||^2)
 */
inline fvalue CGaussKernel::m_evaluateKernel(fvalue euclideanDistanceSquared) {
	return exp(m_negativeGamma * euclideanDistanceSquared);
}


template<class Matrix>
class RbfKernelEvaluator {

private:
	Matrix* samples;
	label_id* labels;
	fvalue c;

	fvalue yyNeg;

	fvalue betta;
	fvalue bias;

	fvalue epochs;
	fvalue margin;

protected:
  CGaussKernel params;
	MatrixEvaluator<Matrix> eval;

	fvalue rbf(fvalue dist2);

public:
	RbfKernelEvaluator(Matrix* samples, label_id* labels, quantity classNumber, fvalue bias, fvalue c, CGaussKernel &params, fvalue epochs, fvalue margin);
	~RbfKernelEvaluator();

	void evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector* result);

	void swapSamples(sample_id uid, sample_id vid);
	void setKernelParams(fvalue c, CGaussKernel &params);

  CGaussKernel getParams();
	fvalue getC();
	fvalue getBias();
	fvalue getBetta();
	fvalue getEpochs();
	fvalue getMargin();
	void updateBias(fvalue LB);
	void resetBias();

	fvalue getLabel(sample_id v);
	void setLabel(sample_id v);
};

template<class Matrix>
RbfKernelEvaluator<Matrix>::RbfKernelEvaluator(Matrix* samples, label_id* labels, quantity classNumber,
		fvalue betta, fvalue c, CGaussKernel &params, fvalue epochs, fvalue margin) :
		samples(samples),
		labels(labels),
		c(c),
		betta(betta),
		params(params),
		eval(samples),
		epochs(epochs),
		margin(margin) {
	bias = 0.0;
	yyNeg = -1.0 / (classNumber - 1);
}

template<class Matrix>
RbfKernelEvaluator<Matrix>::~RbfKernelEvaluator() {
}

/*
 * Updates the current model's bias with the appropriate gradient value (learning rate * C * label / currentSize)
 */
template<typename Matrix>
inline void RbfKernelEvaluator<Matrix>::updateBias(fvalue biasGradient) {
	bias = bias + biasGradient;
}

/*
 * Reset's the evaluators bias value to 0. This is needed when creating a new pairwise model.
 */
template<typename Matrix>
inline void RbfKernelEvaluator<Matrix>::resetBias() {
	bias = 0.0;
}

/*
* Sets the training pair comparison value to be the second.
*/
template<typename Matrix>
inline void RbfKernelEvaluator<Matrix>::setLabel(sample_id v) {
	yyNeg = v;
}

/*
* Returns the label (+1 or -1)
*/
template<typename Matrix>
inline fvalue RbfKernelEvaluator<Matrix>::getLabel(sample_id v) {
	fvalue vals[] = { 1.0, -1.0 }; //TODO: fix this to be the following: 2*label[v] - 1
	fvalue label = vals[labels[v] == yyNeg];
	return label;
}

/*
 * Calculated the Gaussian RBF kernel value given a euclidean distance squared between two samples.
 */
template<class Matrix>
inline fvalue RbfKernelEvaluator<Matrix>::rbf(fvalue euclideanDistanceSquared) {
	return params.m_evaluateKernel(euclideanDistanceSquared);
}

template<class Matrix>
void RbfKernelEvaluator<Matrix>::evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector* result) 
{
	eval.dist(id, rangeFrom, rangeTo, result);

	fvalue* rptr = fvector_ptr(result);
	for (sample_id iid = rangeFrom; iid < rangeTo; iid++) {
		rptr[iid] = rbf(rptr[iid]);
	}
}

template<class Matrix>
inline void RbfKernelEvaluator<Matrix>::swapSamples(sample_id uid, sample_id vid) {
	swap(labels[uid], labels[vid]);
	eval.swapSamples(uid, vid);
}

template<class Matrix>
void RbfKernelEvaluator<Matrix>::setKernelParams(fvalue c, CGaussKernel &params) {
	this->c = c;
	this->params = params;
}

template<class Matrix>
inline CGaussKernel RbfKernelEvaluator<Matrix>::getParams() {
	return params;
}

template<class Matrix>
inline fvalue RbfKernelEvaluator<Matrix>::getC() {
	return c;
}

template<class Matrix>
inline fvalue RbfKernelEvaluator<Matrix>::getBias() {
	return bias;
}

template<class Matrix>
inline fvalue RbfKernelEvaluator<Matrix>::getBetta() {
	return betta;
}

template<class Matrix>
inline fvalue RbfKernelEvaluator<Matrix>::getEpochs () {
	return epochs;
}

template<class Matrix>
inline fvalue RbfKernelEvaluator<Matrix>::getMargin() {
	return margin;
}

#endif
