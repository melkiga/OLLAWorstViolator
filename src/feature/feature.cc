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

#include "feature.h"

void FeatureProcessor::normalize(sfmatrix *samples) {
	// project features to the range [0, 1]
	vector<fvalue> maxs(samples->width, 0.0);
	for (sample_id smpl = 0; smpl < samples->height; smpl++) {
		id offset = samples->offsets[smpl];
		while (samples->features[offset] != INVALID_FEATURE_ID) {
			feature_id feature = samples->features[offset];
			fvalue value = samples->values[offset];
			maxs[feature] = max(maxs[feature], (fvalue) fabs(value));
			offset++;
		}
	}
	for (sample_id smpl = 0; smpl < samples->height; smpl++) {
		id offset = samples->offsets[smpl];
		while (samples->features[offset] != INVALID_FEATURE_ID) {
			feature_id feature = samples->features[offset];
			samples->values[offset] /= maxs[feature];
			offset++;
		}
	}
}

void FeatureProcessor::randomize(sfmatrix *samples, label_id *labels) {
	IdGenerator gen = Generators::create();
	for (id row = 0; row < samples->height; row++) {
		id rand = gen.nextId(samples->height);
		swap(samples->offsets[row], samples->offsets[rand]);
		swap(labels[row], labels[rand]);
	}
}
