#include "validation.h"

CrossValidationSolver::CrossValidationSolver(AbstractSolver *solver, quantity innerFolds, quantity outerFolds, bool fairFoolds) :
		solver(solver),
		innerFoldsNumber(innerFolds),
		outerFoldsNumber(outerFolds) {
	quantity size = solver->getSize();
	label_id *labels = solver->getLabels();
	map<label_id, string> &labelNames = solver->getLabelNames();

	innerFoldsMembership = new fold_id[size];
	innerFoldSizes = new quantity[innerFolds];
	for (id i = 0; i < innerFolds; i++) {
		innerFoldSizes[i] = size;
	}
	outerFoldsMembership = new fold_id[size];
	outerFoldSizes = new quantity[outerFolds];
	for (id i = 0; i < outerFolds; i++) {
		outerFoldSizes[i] = size;
	}

	if (fairFoolds) {
		quantity labelNum = (quantity) labelNames.size();
		quantity foldNum = innerFolds * outerFolds;
		id *offsets = new id[labelNum];
		quantity step = innerFolds + 1;
		quantity increase = max(foldNum / labelNum, (quantity) 1);
		for (quantity i = 0; i < labelNum; i++) {
			offsets[i] = (i * increase * step) % foldNum;
		}
		for (id i = 0; i < size; i++) {
			label_id label = labels[i];

			id inner = offsets[label] % innerFolds;
			innerFoldsMembership[i] = inner;
			innerFoldSizes[inner]--;

			id outer = offsets[label] / innerFolds;
			outerFoldsMembership[i] = outer;
			if (outerFolds > 1) {
				outerFoldSizes[outer]--;
			}

			offsets[label] = (offsets[label] + step) % foldNum;
		}
		delete [] offsets;
	} else {
		IdGenerator innerGen = Generators::create();
		for (id i = 0; i < size; i++) {
			id genId = innerGen.nextId(innerFolds);
			innerFoldsMembership[i] = genId;
			innerFoldSizes[genId]--;
		}

		IdGenerator outerGen = Generators::create();
		for (id i = 0; i < size; i++) {
			id genId = outerGen.nextId(outerFolds);
			outerFoldsMembership[i] = genId;
			if (outerFolds > 1) {
				outerFoldSizes[genId]--;
			}
		}
	}

	outerFold = 0;
	outerProblemSize = size;

	SwapListener *listener = new CrossSolverSwapListener(
			innerFoldsMembership, outerFoldsMembership);
	solver->setSwapListener(listener);
}

CrossValidationSolver::~CrossValidationSolver() {
	delete solver;
	delete [] innerFoldsMembership;
	delete [] outerFoldsMembership;
	delete [] innerFoldSizes;
	delete [] outerFoldSizes;
}

void CrossValidationSolver::sortVectors(fold_id *membership, fold_id fold, quantity num) {
	id train = 0;
	id test = num - 1;
	while (train <= test) {
		while (train < num && membership[train] != fold) {
			train++;
		}
		while (test >= 0 && membership[test] == fold) {
			test--;
		}
		if (train < test) {
			solver->swapSamples(train, test);
			train++;
			test--;
		}
	}
}

void CrossValidationSolver::resetInnerFold(fold_id fold) {
#ifdef SUPPORT_VECTOR_REUSE
	solver->shrink();
	sortVectors(this->innerFoldsMembership, fold, outerFoldSizes[outerFold]);
#else
	sortVectors(this->innerFoldsMembership, fold, outerFoldSizes[outerFold]);
	solver->reset();
#endif

	solver->setCurrentSize(innerFoldSizes[fold]);
}

void CrossValidationSolver::resetOuterFold(fold_id fold) {
	outerFold = fold;

	sortVectors(this->outerFoldsMembership, fold, solver->getSize());

	for (id i = 0; i < innerFoldsNumber; i++) {
		innerFoldSizes[i] = outerFoldSizes[fold];
	}
	for (id i = 0; i < outerFoldSizes[fold]; i++) {
		id id = innerFoldsMembership[i];
		innerFoldSizes[id]--;
	}

	outerProblemSize = outerFoldSizes[fold];
	solver->reset();
}

void CrossValidationSolver::trainOuter() {
	solver->reset();

	solver->setCurrentSize(outerFoldSizes[outerFold]);

	this->train();
}

TestingResult CrossValidationSolver::test(sample_id from, sample_id to) {
	label_id *labels = solver->getLabels();
	quantity correct = 0;
	Classifier *classifier = solver->getClassifier();
	for (sample_id test = from; test < to; test++) {
		label_id label = classifier->classify(test);
		if (label == labels[test]) {
			correct++;
		}
	}
	delete classifier;
	return TestingResult((fvalue) correct / (to - from));
}

TestingResult CrossValidationSolver::doCrossValidation() {
	TestingResult result;

	Timer timer;

	for (fold_id innerFold = 0; innerFold < innerFoldsNumber; innerFold++) {
		resetInnerFold(innerFold);

		timer.restart();
		solver->train();
		timer.stop();

		logger << format("inner fold %d/%d training: time=%.2f[s], sv=%d/%d\n")
				% outerFold % innerFold % timer.getTimeElapsed()
				% solver->getSvNumber() % solver->getCurrentSize();

		timer.restart();
		TestingResult foldResult = test(
				this->innerFoldSizes[innerFold], this->outerFoldSizes[outerFold]);
		timer.stop();

		logger << format("inner fold %d/%d testing: time=%.2f[s], accuracy=%.2f[%%]\n")
				% outerFold % innerFold
				% timer.getTimeElapsed() % (100.0 * foldResult.accuracy);

		result.accuracy += foldResult.accuracy / innerFoldsNumber;
	}

	return result;
}


quantity CrossValidationSolver::getInnerFoldsNumber() {
	return innerFoldsNumber;
}


quantity CrossValidationSolver::getOuterFoldsNumber() {
	return outerFoldsNumber;
}


fold_id CrossValidationSolver::getOuterFold() {
	return outerFold;
}


quantity CrossValidationSolver::getOuterProblemSize() {
	return outerProblemSize;
}
