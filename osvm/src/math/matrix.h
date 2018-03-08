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

#ifndef MATRIX_H_
#define MATRIX_H_

#include <algorithm>

#include "numeric.h"
#include "matrix_sparse.h"
#include "matrix_dense.h"

template<typename Matrix>
class EvaluatorWorkspace {

};


template<>
class EvaluatorWorkspace<dfmatrix> {
public:
	fvectorv urow;
	fvectorv vrow;

	fmatrixv matrixView;
	fvectorv rowView;
	fvectorv bufferView;
	fvectorv x2View;

	fvalue* matrixData;
	size_t matrixTda;

	sample_id* forwardMap;
	sample_id* reverseMap;

	EvaluatorWorkspace(dfmatrix* matrix);
	~EvaluatorWorkspace();

};


template<>
class EvaluatorWorkspace<sfmatrix> {
public:
	fvalue* buffer;

	EvaluatorWorkspace(sfmatrix* matrix);
	~EvaluatorWorkspace();

};


/*
 * Structure for holding the data (sparsely or densely) and other relevant information for kernel calculations.
 */
template<typename Matrix>
class MatrixEvaluator {

	Matrix* matrix; // this would house the samples
	fvalue* x2; // ||x||^2 (this would be the squared 2-norm of a sample x)
	EvaluatorWorkspace<Matrix> workspace;

protected:
	quantity getSize(Matrix* matrix);
	quantity getDim(Matrix* matrix);

	fvalue squaredNorm(sample_id v);

public:
	MatrixEvaluator(Matrix* matrix);
	~MatrixEvaluator();

	fvalue dot(sample_id u, sample_id v);

	void dist(sample_id id, sample_id rangeFrom, sample_id rangeTo,
			fvector* buffer);
	void dist(sample_id id, sample_id rangeFrom, sample_id rangeTo,
			sample_id* mappings, fvector* buffer);
	fvalue dist(sample_id u, sample_id v);

	void swapSamples(sample_id u, sample_id v);

};

template<typename Matrix>
MatrixEvaluator<Matrix>::MatrixEvaluator(Matrix* matrix) :
		matrix(matrix),
		workspace(matrix) {
	quantity length = getSize(matrix);
	x2 = new fvalue[length];
	for (sample_id id = 0; id < length; id++) {
		x2[id] = squaredNorm(id);
	}
}

template<typename Matrix>
MatrixEvaluator<Matrix>::~MatrixEvaluator() {
	delete [] x2;
}

template<typename Matrix>
void MatrixEvaluator<Matrix>::dist(sample_id id, sample_id rangeFrom, sample_id rangeTo,
		sample_id* mappings, fvector* buffer) {
	fvalue* data = buffer->data;
	for (sample_id i = rangeFrom; i < rangeTo; i++) {
		data[i] = dist(mappings[i], id);
	}
}

/*
 * For samples u and v, this calculated the euclidean distance between the two.
 * returns: ||u - v||^2: ||u||^2 + ||v||^2 - 2*(<u,v>)
 */
template<typename Matrix>
fvalue MatrixEvaluator<Matrix>::dist(sample_id u, sample_id v) {
	return x2[u] + x2[v] - 2 * dot(u, v);
}

/*
 * Returns the number of samples of the data (aka matrix height)
 */
template<typename Matrix>
inline quantity MatrixEvaluator<Matrix>::getSize(Matrix* matrix) {
	return (quantity) matrix->height;
}

/*
* Returns the dimensionality of the data (number of attributes)
*/
template<typename Matrix>
inline quantity MatrixEvaluator<Matrix>::getDim(Matrix* matrix) {
	return (quantity) matrix->width;
}

#endif
