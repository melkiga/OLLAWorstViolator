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

#ifndef GENERATOR_H_
#define GENERATOR_H_

#include <algorithm>
#include "../math/numeric.h"
#include "../math/random.h"
#include "params.h"


enum GeneratorType {
	FAIR
};

template<GeneratorType Type>
class CandidateIdGenerator {

public:
	CandidateIdGenerator(TrainParams &params, quantity labelNumber, label_id *labels, quantity sampleNumber);

	id nextId();

	void exchange(id u, id v);

};

struct IdNode {

	id value;
	id next;
	quantity succeeded;
	quantity failed;

	IdNode() : value(INVALID_ID), next(INVALID_ID), succeeded(1), failed(0) {
	}

};

struct ClassDistribution {

	quantity maxLabelNumber;
	quantity labelNumber;
	vector<id> labelMappings;
	vector<quantity> bufferSizes;
	vector<id*> buffers;
	vector<id> bufferHolder;
	vector<id> offsets;

	ClassDistribution(quantity labelNum, label_id *smplMemb, quantity smplNum);
	~ClassDistribution();

	void refresh(label_id *smplMemb, quantity smplNum);
	void exchange(sample_id u, sample_id v);

};

template<>
class CandidateIdGenerator<FAIR> {

	IdGenerator generator;
	ClassDistribution distr;

public:
	CandidateIdGenerator(TrainParams &params, quantity labelNumber, label_id *labels, quantity sampleNumber);

	id nextId();

	void exchange(id u, id v);
	void reset(label_id *labels, id maxId);

};

inline CandidateIdGenerator<FAIR>::CandidateIdGenerator(TrainParams& params,
		quantity labelNumber, label_id *labels, quantity sampleNumber) :
		generator(Generators::create()),
		distr(ClassDistribution(labelNumber, labels, sampleNumber)) {
}

inline id CandidateIdGenerator<FAIR>::nextId() {
	label_id label = distr.labelMappings[generator.nextId(distr.labelNumber)];
	id offset = generator.nextId(distr.bufferSizes[label]);
	return distr.buffers[label][offset];
}

inline void CandidateIdGenerator<FAIR>::exchange(id u, id v) {
	distr.exchange(u, v);
}

inline void CandidateIdGenerator<FAIR>::reset(label_id *labels, id maxId) {
	distr.refresh(labels, maxId);
}

#endif
