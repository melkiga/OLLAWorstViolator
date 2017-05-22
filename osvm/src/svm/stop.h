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

#ifndef STOP_H_
#define STOP_H_

#include "../math/numeric.h"

#define DEFAULT_EPSILON 0.1

#define EXPECTED_SV_NUMBER 15000


class StopCriterionStrategy {

public:
	virtual fvalue getThreshold(fvalue eps, fvalue c) = 0;
	virtual fvalue getEpsilon() = 0;

	virtual ~StopCriterionStrategy();

};


class L1SVMStopStrategy: public StopCriterionStrategy {

public:
	virtual fvalue getThreshold(fvalue eps, fvalue c);
	virtual fvalue getEpsilon();

	virtual ~L1SVMStopStrategy();

};


#endif
