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

#ifndef PARAMS_H_
#define PARAMS_H_

#include "../math/numeric.h"

#define DEFAULT_ETA 1.0
#define DEFAULT_DRAW_NUMBER 590
#define DEFAULT_EPOCHS 0.5
#define DEFAULT_MARGIN 0.1

#define DEFAULT_CACHE_SIZE 200

#define DEFAULT_STOPPING_L1SVM_K 1.0

#define DEFAULT_GENERATOR_BUCKET_NUMBER 1025

enum BiasType {
	NO,
	YES
};

struct TrainParams {
	quantity drawNumber;

	BiasType bias;
	fvalue epochs;
	fvalue margin;

	struct {
		quantity size;
	} cache;

	struct {
		fvalue k;
	} stopping;

	struct {
		// deterministic
		quantity bucketNumber;
	} generator;

	TrainParams();

};

#endif
