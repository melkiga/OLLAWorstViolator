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

#ifndef UPDATE_H_
#define UPDATE_H_

#include "generator.h"
#include "params.h"

template<GeneratorType Generator>
class SolverStrategy {

	CandidateIdGenerator<Generator> generator;

public:
	SolverStrategy(TrainParams &params, quantity labelNumber, label_id *labels, quantity sampleNumber);

	void resetGenerator(label_id *labels, id maxId);
	void notifyExchange(id u, id v);

};

template<GeneratorType Generator>
inline SolverStrategy<Generator>::SolverStrategy(TrainParams& params,
		quantity labelNumber, label_id *labels, quantity sampleNumber) :
		generator(params, labelNumber, labels, sampleNumber) {
}

template<GeneratorType Generator>
inline void SolverStrategy<Generator>::resetGenerator(label_id *labels, id maxId) {
	generator.reset(labels, maxId);
}

template<GeneratorType Generator>
inline void SolverStrategy<Generator>::notifyExchange(id u, id v) {
	generator.exchange(u, v);
}

#endif
