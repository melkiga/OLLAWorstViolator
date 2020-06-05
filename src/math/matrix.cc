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

#include "matrix.h"

EvaluatorWorkspace::EvaluatorWorkspace(sfmatrix *matrix) {
	buffer = new fvalue[matrix->width];
	for (size_t i = 0; i < matrix->width; i++) {
		buffer[i] = 0;
	}
}

EvaluatorWorkspace::~EvaluatorWorkspace() {
	delete [] buffer;
}

MatrixEvaluator::MatrixEvaluator(sfmatrix* matrix) :
  matrix(matrix),
  workspace(matrix) {
  quantity length = getSize(matrix);
  x2 = new fvalue[length];
  for (sample_id id = 0; id < length; id++) {
    x2[id] = squaredNorm(id);
  }
}

MatrixEvaluator::~MatrixEvaluator() {
  delete[] x2;
}

void MatrixEvaluator::dist(sample_id id, sample_id rangeFrom, sample_id rangeTo, sample_id* mappings, fvector* buffer) {
  fvalue* data = buffer->data;
  for (sample_id i = rangeFrom; i < rangeTo; i++) {
    data[i] = dist(mappings[i], id);
  }
}

/*
* For samples u and v, this calculated the euclidean distance between the two.
* returns: ||u - v||^2: ||u||^2 + ||v||^2 - 2*(<u,v>)
*/
fvalue MatrixEvaluator::dist(sample_id u, sample_id v) {
  return x2[u] + x2[v] - 2 * dot(u, v);
}

fvalue MatrixEvaluator::dot(sample_id u, sample_id v) {
	id uoffset = matrix->offsets[u];
	feature_id *iuptr = matrix->features + uoffset;
	fvalue *fuptr = matrix->values + uoffset;

	id voffset = matrix->offsets[v];
	feature_id *ivptr = matrix->features + voffset;
	fvalue *fvptr = matrix->values + voffset;

	fvalue sum = 0.0;
	while (*iuptr != INVALID_FEATURE_ID && *ivptr != INVALID_FEATURE_ID) {
		while (*iuptr < *ivptr) {
			iuptr++;
			fuptr++;
		}
		if (*iuptr == *ivptr) {
			sum += *fuptr * *fvptr;
			iuptr++;
			fuptr++;
		}
		swap(iuptr, ivptr);
		swap(fuptr, fvptr);
	}
	return sum;
}

/*
* Calculates the squared norm of sample u for samples stored in sparse format
*/
fvalue MatrixEvaluator::squaredNorm(sample_id u) {
	fvalue sum = 0.0;
	id offset = matrix->offsets[u];
	feature_id *iptr = matrix->features + offset;
	fvalue *fptr = matrix->values + offset;
	while (*iptr++ != INVALID_FEATURE_ID) {
		sum += pow2(*fptr++);
	}
	return sum;
}

void MatrixEvaluator::dist(sample_id v, sample_id rangeFrom, sample_id rangeTo, fvector *buffer) {
	// setup workspace
	id offset = matrix->offsets[v];
	feature_id *iptr = matrix->features + offset;
	fvalue *fptr = matrix->values + offset;
	while (*iptr != INVALID_FEATURE_ID) {
		workspace.buffer[*iptr++] = *fptr++;
	}

	// calculate dists
	fvalue *x2ptr = x2;
	fvalue v2 = x2ptr[v];
	fvalue *fbuffer = buffer->data;
	for (sample_id offst = rangeFrom; offst < rangeTo; offst++) {
		id coffset = matrix->offsets[offst];
		feature_id *icptr = matrix->features + coffset;
		fvalue *fcptr = matrix->values + coffset;
		fvalue sum = 0.0;
		while (*icptr != INVALID_FEATURE_ID) {
			sum += *fcptr++ * workspace.buffer[*icptr++];
		}
		fbuffer[offst] = x2ptr[offst] + v2 - 2.0 * sum;
	}

	// clear workspace
	iptr = matrix->features + offset;
	while (*iptr != INVALID_FEATURE_ID) {
		workspace.buffer[*iptr++] = 0.0;
	}
}

void MatrixEvaluator::swapSamples(sample_id u, sample_id v) {
	swap(matrix->offsets[u], matrix->offsets[v]);
	swap(x2[u], x2[v]);
}
