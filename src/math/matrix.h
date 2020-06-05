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

class EvaluatorWorkspace {
public:
	fvalue* buffer;

	EvaluatorWorkspace(sfmatrix* matrix);
	~EvaluatorWorkspace();
};


/*
 * Structure for holding the data (sparsely or densely) and other relevant information for kernel calculations.
 */
class MatrixEvaluator {

	sfmatrix* matrix; // this would house the samples
	fvalue* x2; // ||x||^2 (this would be the squared 2-norm of a sample x)
	EvaluatorWorkspace workspace;

protected:
	quantity getSize(sfmatrix* matrix);
	quantity getDim(sfmatrix* matrix);

	fvalue squaredNorm(sample_id v);

public:
	MatrixEvaluator(sfmatrix* matrix);
	~MatrixEvaluator();

	fvalue dot(sample_id u, sample_id v);

	void dist(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector* buffer);
	void dist(sample_id id, sample_id rangeFrom, sample_id rangeTo, sample_id* mappings, fvector* buffer);
	fvalue dist(sample_id u, sample_id v);

	void swapSamples(sample_id u, sample_id v);

};

/*
 * Returns the number of samples of the data (aka matrix height)
 */
inline quantity MatrixEvaluator::getSize(sfmatrix* matrix) {
	return (quantity) matrix->height;
}

/*
* Returns the dimensionality of the data (number of attributes)
*/
inline quantity MatrixEvaluator::getDim(sfmatrix* matrix) {
	return (quantity) matrix->width;
}

#endif
