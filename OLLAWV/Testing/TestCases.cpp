#include "TestCases.h"
#include "../../osvm/src/launcher.h"

using namespace std;


// ------------------------------------------------------------------------------------------------------------------------------
// Some kind of an eval function...

// ------------------------------------------------------------------------------------------------------------------------------
// "Entry point" for test case runner.
int TestCaseMain()
{
  InitOptions();



  // Use a relative path for the input file.
  char* input = "-i 1 -o 1 \"..\\TestData\\small-data\\iris\"";

  try
  {
    // Run the code.
    Configuration conf = GetConfiguration(input);
    ApplicationLauncher launcher(conf);
    PairwiseClassifier* c = (PairwiseClassifier*)launcher.run();
    PairwiseTrainingResult* result = c->getState();

    
    // Evaluate post-conditions.
    size_t modelCount = result->models.size();
    Check::AreEqual<quantity>(0, modelCount, "There should be zero models!");

    // @@AAR: 6.12.2018.
    // The following two conditions are failing because they are accessing data that
    // is uninitialized.  BAD!  That would be something we would want to get updated.
    quantity labelCount = result->totalLabelCount;
    Check::AreEqual<quantity>(0, labelCount);

    quantity maxSVCount = result->maxSVCount;
    Check::AreEqual<quantity>(0, maxSVCount);

    cout << "The test case passed:";

  }
  catch (const std::exception& ex)
  {
    // Make note of the error.
    cout << "The test case failed because: \r\n" << ex.what();
    int x = 10;
  }


  int x;
  cin >> x;

  return 0;
}
