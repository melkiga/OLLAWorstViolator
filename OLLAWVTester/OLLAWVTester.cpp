#include "stdafx.h"

#include "OLLAWVTester.h"
//#include "OSVMDefs.h"

using namespace std;

// This fixes the legacy iob linker error stuff....
extern "C" { FILE __iob_func[3] = { *stdin,*stdout,*stderr }; }

int main()
{

  // This part was cribbed from osvm.cc
  // This is how it deals with its command line stuff.
  string usage = (format("Usage: %s [OPTION]... [FILE]\n") % PACKAGE).str();
  string descr = "Perform SVM training for the given data set [FILE].\n";
  string options = "Available options";
  options_description desc(usage + descr + options);
  desc.add_options()
    (PR_HELP, "produce help message")
    (PR_C_LOW, value<fvalue>()->default_value(0.001), "C value (lower bound)")
    (PR_C_HIGH, value<fvalue>()->default_value(10000.0), "C value (upper bound)")
    (PR_G_LOW, value<fvalue>()->default_value(0.0009765625), "gamma value (lower bound)")
    (PR_G_HIGH, value<fvalue>()->default_value(16.0), "gamma value (upper bound)")
    (PR_RES, value<int>()->default_value(8), "resolution (for C and gamma)")
    (PR_OUTER_FLD, value<int>()->default_value(1), "outer folds")
    (PR_INNER_FLD, value<int>()->default_value(10), "inner folds")
    (PR_BIAS_CALCULATION, value<string>()->default_value(BIAS_CALCULATION_YES), "bias evaluation strategy (yes, no)")
    (PR_DRAW_NUM, value<int>()->default_value(600), "draw number")
    (PR_MULTICLASS, value<string>()->default_value(MULTICLASS_ALL_AT_ONCE), "multiclass training approach (allatonce or pairwise)")
    (PR_SEL_TYPE, value<string>()->default_value(SEL_TYPE_PATTERN), "model selection type (grid or pattern)")
    (PR_MATRIX_TYPE, value<string>()->default_value(MAT_TYPE_SPARSE), "data representation (sparse or dense)")
    (PR_STOP_CRIT, value<string>()->default_value(STOP_CRIT_YOC), "stopping criterion (yoC)")
    (PR_OPTIMIZATION, value<string>()->default_value(OPTIMIZATION_L1SVM), "optimization strategy (mdm, imdm, gmdm)")
    (PR_ID_RANDOMIZER, value<string>()->default_value(ID_RANDOMIZER_FAIR), "id generator (simple, fair or determ)")
    (PR_CACHE_SIZE, value<int>()->default_value(DEFAULT_CACHE_SIZE), "cache size (in MB)")
    (PR_EPOCH, value<fvalue>()->default_value(0.5), "epochs number")
    (PR_MARGIN, value<fvalue>()->default_value(0.1), "margin")
    (PR_INPUT, value<string>(), "input file");

  positional_options_description opt;
  opt.add(PR_KEY_INPUT, -1);


  // We need to parse some of these out I think....
  // The first arg is always going to be the program name.  The rest of it is space delimited (except in quotes)
  int argc = 0;
  char** argv = ParseCommand("-u pairwise -i 5 -o 5 \"C:\\data\\smalldata\\iris\"", argc);

  auto parser = command_line_parser(argc, argv);
  auto parsed = parser.options(desc).positional(opt).run();
  variables_map vars;
  store(parsed, vars);
  notify(vars);

  cout << "We parsed a command line!  Yay!";


  ParametersParser pp(vars);
  Configuration conf; // = pp.getConfiguration();

  return 0;
}


//
//#ifdef ENGINE_LIB
//#define ENGINE_API __declspec(dllexport)
//#else
//#define ENGINE_API __declspec(dllimport) 
//#endif
