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

struct GaussKernel {
	fvalue ngamma;

	GaussKernel(fvalue gamma);

	fvalue eval(fvalue dist2);

};

inline fvalue GaussKernel::eval(fvalue dist2) {
	return exp(ngamma * dist2);
}


template<class Kernel, class Matrix>
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
	Kernel params;
	MatrixEvaluator<Matrix> eval;

	fvalue rbf(fvalue dist2);

public:
	RbfKernelEvaluator(Matrix* samples, label_id* labels, quantity classNumber, fvalue bias, fvalue c, Kernel &params, fvalue epochs, fvalue margin);
	~RbfKernelEvaluator();

	fvalue evalInnerKernel(sample_id uid, sample_id vid);
	void evalInnerKernel(sample_id id, sample_id rangeFrom,
			sample_id rangeTo, fvector* result);
	void evalInnerKernel(sample_id id, sample_id rangeFrom,
			sample_id rangeTo, sample_id* mappings, fvector* result);

	fvalue evalKernel(sample_id uid, sample_id vid);
	void evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector* result);

	void swapSamples(sample_id uid, sample_id vid);
	void setKernelParams(fvalue c, Kernel &params);

	Kernel getParams();
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

template<class Kernel, class Matrix>
RbfKernelEvaluator<Kernel, Matrix>::RbfKernelEvaluator(Matrix* samples, label_id* labels, quantity classNumber,
		fvalue betta, fvalue c, Kernel &params, fvalue epochs, fvalue margin) :
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

template<class Kernel, class Matrix>
RbfKernelEvaluator<Kernel, Matrix>::~RbfKernelEvaluator() {
}

template<typename Kernel, typename Matrix>
inline void RbfKernelEvaluator<Kernel, Matrix>::updateBias(fvalue LB) {
	bias = bias + LB;
}

template<typename Kernel, typename Matrix>
inline void RbfKernelEvaluator<Kernel, Matrix>::resetBias() {
	bias = 0.0;
}

/*
* Sets the training pair comparison value to be the second. TODO: explain more.
*/
template<typename Kernel, typename Matrix>
inline void RbfKernelEvaluator<Kernel, Matrix>::setLabel(sample_id v) {
	yyNeg = v;
}

/*
* Returns the label (+1 or -1)
*/
template<typename Kernel, typename Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::getLabel(sample_id v) {
	fvalue vals[] = { 1.0, -1.0 };
	fvalue label = vals[labels[v] == yyNeg];
	return label;
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::rbf(fvalue dist2) {
	return params.eval(dist2);
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::evalInnerKernel(
		sample_id uid, sample_id vid) {
	return rbf(eval.dist(uid, vid));
}

template<class Kernel, class Matrix>
void RbfKernelEvaluator<Kernel, Matrix>::evalInnerKernel(sample_id id,
		sample_id rangeFrom, sample_id rangeTo, fvector* result) {
	eval.dist(id, rangeFrom, rangeTo, result);

	fvalue* ptr = fvector_ptr(result);
	for (sample_id iid = rangeFrom; iid < rangeTo; iid++) {
		ptr[iid] = rbf(ptr[iid]);
	}
}

template<class Kernel, class Matrix>
void RbfKernelEvaluator<Kernel, Matrix>::evalInnerKernel(sample_id id,
		sample_id rangeFrom, sample_id rangeTo,
		sample_id* mappings, fvector* result) {
	eval.dist(id, rangeFrom, rangeTo, mappings, result);

	fvalue* ptr = fvector_ptr(result);
	for (sample_id iid = rangeFrom; iid < rangeTo; iid++) {
		ptr[iid] = rbf(ptr[iid]);
	}
}

/*
 * Evaluates the RBF kernel between 2 samples.
 */
// TODO: change evalKernel(id1,id2) to remove yyNeg and tau
template<class Kernel, class Matrix>
fvalue RbfKernelEvaluator<Kernel, Matrix>::evalKernel(sample_id uid, sample_id vid) {
	label_id ulabel = labels[uid];
	label_id vlabel = labels[vid];
	fvalue result;
	if (ulabel == vlabel) {
		if (uid == vid) {
			result = tau;
		} else {
			result = evalInnerKernel(uid, vid) + bias;
		}
	} else {
		result = yyNeg * (evalInnerKernel(uid, vid) + bias);
	}
	return result;
}

template<class Kernel, class Matrix>
void RbfKernelEvaluator<Kernel, Matrix>::evalKernel(sample_id id,
		sample_id rangeFrom, sample_id rangeTo, fvector* result) {
	eval.dist(id, rangeFrom, rangeTo, result);

	fvalue* rptr = fvector_ptr(result);
	for (sample_id iid = rangeFrom; iid < rangeTo; iid++) {
		rptr[iid] = rbf(rptr[iid]);
	}
}

template<class Kernel, class Matrix>
inline void RbfKernelEvaluator<Kernel, Matrix>::swapSamples(sample_id uid, sample_id vid) {
	swap(labels[uid], labels[vid]);
	eval.swapSamples(uid, vid);
}

template<class Kernel, class Matrix>
void RbfKernelEvaluator<Kernel, Matrix>::setKernelParams(fvalue c, Kernel &params) {
	this->c = c;
	this->params = params;
}

template<class Kernel, class Matrix>
inline Kernel RbfKernelEvaluator<Kernel, Matrix>::getParams() {
	return params;
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::getC() {
	return c;
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::getBias() {
	return bias;
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::getBetta() {
	return betta;
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::getEpochs () {
	return epochs;
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::getMargin() {
	return margin;
}

#endif
