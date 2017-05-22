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

#ifndef VIOLATION_H_
#define VIOLATION_H_

#include "../math/numeric.h"
#include "params.h"

/// <summary>For Optimization. Enum stating the minimal norm problem solving strategy.</summary>
enum ViolationCriterion {
	L1SVM
};

template<ViolationCriterion Type>
class ViolationEstimator {

public:
	ViolationEstimator(TrainParams &params);

};

template<>
class ViolationEstimator<L1SVM> {

	fvalue k;

public:
	ViolationEstimator(TrainParams &params);

};


template<ViolationCriterion Type>
inline ViolationEstimator<Type>::ViolationEstimator(TrainParams& params) {
}


inline ViolationEstimator<L1SVM>::ViolationEstimator(TrainParams& params) {
	k = params.stopping.k;
}


#endif
