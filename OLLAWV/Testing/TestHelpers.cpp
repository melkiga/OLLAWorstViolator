#include "TestHelpers.h"


string usage = (format("Usage: %s [OPTION]... [FILE]\n") % PACKAGE).str();
string descr = "Perform SVM training for the given data set [FILE].\n";
string options = "Available options";
options_description InputOptions(usage + descr + options);

// ------------------------------------------------------------------------------------------------------------------------------
void InitOptions()
{
  // Get the options ready.
  InputOptions.add_options()
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
}

// ------------------------------------------------------------------------------------------------------------------------------
Configuration GetConfiguration(char* input)
{

  // Do I really have to make a new one of these every time.... -__-

  int argc = 0;
  char** argv = ParseCommand(input, argc);

  positional_options_description opt;
  opt.add(PR_KEY_INPUT, -1);

  variables_map vars;
  store(command_line_parser(argc, argv).options(InputOptions).positional(opt).run(), vars);
  notify(vars);

  CleanupCommandArgs(argv, argc);

  if (!vars.count(PR_KEY_HELP)) {
    ParametersParser parser(vars);
    Configuration conf = parser.getConfiguration();
    return conf;
  }
  else
  {
    // This is some kind of problem.
    throw std::exception("Could not parse the command line!");
  }
}

// ------------------------------------------------------------------------------------------------------------------------------
char* CreateArgBuffer()
{
  char* res = new char[MAX_SIZE + 1];
  memset(res, 0, MAX_SIZE + 1);

  return res;
}

// ------------------------------------------------------------------------------------------------------------------------------
char** ParseCommand(const char* cmdLine, int& count)
{
  char** res = (char**)calloc(MAX_SIZE, sizeof(char*));

  size_t bufferSize = 0;
  size_t len = strnlen_s(cmdLine, MAX_SIZE);

  // The first argument is the path of the application.  For our test code,
  // this won't matter.
  char* buffer = CreateArgBuffer();
  strcpy_s(buffer, MAX_SIZE, "OLLAWVTester.app");
  res[0] = buffer;
  ++count;

  // Now we can just scan out the input chars from the 'command line'
  buffer = CreateArgBuffer();
  for (size_t i = 0; i < len; i++)
  {
    char next = cmdLine[i];
    if (next == ' ')
    {
      // Thow it on the stack.
      res[count] = buffer;
      ++count;

      // Make a new buffer!
      buffer = CreateArgBuffer();
      bufferSize = 0;
    }
    else
    {
      // We will just reject quotes in our parsing machine.
      if (next != '\"')
      {
        buffer[bufferSize] = next;
        ++bufferSize;
      }
    }
  }

  // Commit the rest of the buffer data.
  if (bufferSize > 0)
  {
    res[count] = buffer;
    ++count;
  }

  return res;
}


// ------------------------------------------------------------------------------------------------------------------------------
// This cleans up the memory from the output of ParseCommand.
void CleanupCommandArgs(char** args, int count)
{
  for (size_t i = 0; i < count; i++)
  {
    delete[](args[i]);
  }
  free(args);
  args = nullptr;
}
