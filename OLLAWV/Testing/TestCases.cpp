#include "TestCases.h"

using namespace std;



// ------------------------------------------------------------------------------------------------------------------------------
// "Entry point" for test case runner.
int TestCaseMain()
{
  InitOptions();

  char* input = "-u pairwise -i 5 -o 5 \"C:\\data\\smalldata\\iris\"";
  Configuration c = GetConfiguration(input);

  return 0;
}
