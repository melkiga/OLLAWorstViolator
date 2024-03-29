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

#ifndef SPARSE_H_
#define SPARSE_H_

#include "numeric.h"

/// <summary>Sparse Matrix Class
/// <param name = "values">...</param>
/// <param name = "features">...</param>
/// <param name = "offsets">...</param>
/// <param name = "height">...</param>
/// <param name = "width">...</param></summary>
struct SparseMatrix {

	fvalue *values;
	feature_id *features;

	id *offsets;
	size_t height;
	size_t width;

	SparseMatrix(fvalue *values, feature_id *features, id *offsets, size_t size1, size_t size2);
	~SparseMatrix();

};

/// Alias for SparseMatrix
typedef SparseMatrix sfmatrix;

#endif
