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

#include "kernel.h"

CGaussKernel::CGaussKernel(fvalue gamma) :
		m_negativeGamma(-gamma) {
}

RbfKernelEvaluator::RbfKernelEvaluator(sfmatrix* samples, label_id* labels, quantity classNumber, fvalue betta, fvalue c, CGaussKernel &params, fvalue epochs, fvalue margin) :
  samples(samples),
  labels(labels),
  c(c),
  betta(betta),
  params(params),
  eval(samples),
  epochs(epochs),
  margin(margin) {
  bias = 0.0;
  yyNeg = -1.0 / (classNumber - 1);
}

RbfKernelEvaluator::~RbfKernelEvaluator() {
}

void RbfKernelEvaluator::evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector* result)
{
  eval.dist(id, rangeFrom, rangeTo, result);

  fvalue* rptr = fvector_ptr(result);
  for (sample_id iid = rangeFrom; iid < rangeTo; iid++) {
    rptr[iid] = rbf(rptr[iid]);
  }
}

void RbfKernelEvaluator::setKernelParams(fvalue c, CGaussKernel &params) {
  this->c = c;
  this->params = params;
}