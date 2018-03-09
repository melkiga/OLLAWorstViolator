#include "TestCases.h"
#include "../../osvm/src/launcher.h"

using namespace std;



// ------------------------------------------------------------------------------------------------------------------------------
// "Entry point" for test case runner.
int TestCaseMain()
{
  InitOptions();



  // Use a relative path for the input file.
  char* input = "-u pairwise -i 1 -o 1 \"..\\TestData\\small-data\\iris\"";

  try
  {
    // Run the code.
    Configuration conf = GetConfiguration(input);
    ApplicationLauncher launcher(conf);
    launcher.launch();

    // Evaluate post-conditions.

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
