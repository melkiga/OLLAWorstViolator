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

#ifndef CV_H_
#define CV_H_

#include "pairwise_solver.h"
#include "../time/timer.h"
#include "../logging/log.h"
#include "../math/random.h"

// comment to disable support vector reuse
//#define SUPPORT_VECTOR_REUSE


struct TestingResult {

	fvalue accuracy;

	TestingResult(fvalue accuracy = 0.0) :
			accuracy(accuracy) {
	}

};


class CrossSolverSwapListener: public SwapListener {

	fold_id *innerMembership;
	fold_id *outerMembership;

public:
	CrossSolverSwapListener(fold_id *innerMembership, fold_id *outerMembership);

	void notify(sample_id u, sample_id v);

};

inline CrossSolverSwapListener::CrossSolverSwapListener(
		fold_id *innerMembership, fold_id *outerMembership) :
		innerMembership(innerMembership),
		outerMembership(outerMembership) {
}

inline void CrossSolverSwapListener::notify(sample_id u, sample_id v) {
	swap(innerMembership[u], innerMembership[v]);
	swap(outerMembership[u], outerMembership[v]);
}


class CrossValidationSolver: public Solver, public DataHolder {

	AbstractSolver *solver;

	fold_id *innerFoldsMembership;
	quantity innerFoldsNumber;
	quantity *innerFoldSizes;

	fold_id *outerFoldsMembership;
	quantity outerFoldsNumber;
	quantity *outerFoldSizes;

	fold_id outerFold;

	quantity outerProblemSize;

protected:
	void resetInnerFold(fold_id fold);

	TestingResult test(sample_id from, sample_id to);
	TestingResult testInner(fold_id fold);

	void sortVectors(fold_id *membership, fold_id fold, quantity num);

public:
	CrossValidationSolver(AbstractSolver *solver,
			quantity innerFolds, quantity outerFolds, bool fairFolds = true);
	virtual ~CrossValidationSolver();

	void setKernelParams(fvalue c, CGaussKernel &params);
	void train();
	Classifier* getClassifier();

	TestingResult doCrossValidation();
	void resetOuterFold(fold_id fold);
	void trainOuter();
	TestingResult testOuter();

	StateHolder& getStateHolder();

	quantity getInnerFoldsNumber();
	quantity getOuterFoldsNumber();
	fold_id getOuterFold();
	quantity getOuterProblemSize();

	sfmatrix* getSamples();
	label_id* getLabels();
	map<label_id, string>& getLabelNames();

};

inline TestingResult CrossValidationSolver::testInner(fold_id fold) {
	return test(this->innerFoldSizes[fold], this->outerFoldSizes[outerFold]);
}


inline TestingResult CrossValidationSolver::testOuter() {
	return test(this->outerFoldSizes[outerFold], solver->getSize());
}

inline sfmatrix* CrossValidationSolver::getSamples() {
	return solver->getSamples();
}


inline label_id* CrossValidationSolver::getLabels() {
	return solver->getLabels();
}


inline map<label_id, string>& CrossValidationSolver::getLabelNames() {
	return solver->getLabelNames();
}


inline StateHolder& CrossValidationSolver::getStateHolder() {
	return *solver;
}


inline void CrossValidationSolver::setKernelParams(fvalue c, CGaussKernel& params) {
	solver->setKernelParams(c, params);
}


inline void CrossValidationSolver::train() {
	solver->train();
}


inline Classifier* CrossValidationSolver::getClassifier() {
	return solver->getClassifier();
}


#endif
