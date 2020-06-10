#include "test_helpers.h"

// program options (command line args)
string usage = (format("Usage: %s [OPTION]... [FILE]\n") % PACKAGE).str();
string descr = "Perform SVM training for the given data set [FILE].\n";
string options = "Available options";
options_description inputOptions(usage + descr + options);

// Get the options ready.
void initOptions(){
  inputOptions.add_options()
    (PR_HELP, "produce help message")
		(PR_C_LOW, value<fvalue>()->default_value(0.001), "C value (lower bound)")
		(PR_C_HIGH, value<fvalue>()->default_value(10000.0), "C value (upper bound)")
		(PR_G_LOW, value<fvalue>()->default_value(0.0009765625), "gamma value (lower bound)")
		(PR_G_HIGH, value<fvalue>()->default_value(16.0), "gamma value (upper bound)")
		(PR_RES, value<int>()->default_value(8), "resolution (for C and gamma)")
		(PR_OUTER_FLD, value<int>()->default_value(1), "outer folds")
		(PR_INNER_FLD, value<int>()->default_value(10), "inner folds")
		(PR_BIAS_CALCULATION, value<string>()->default_value(BIAS_CALCULATION_YES), "bias evaluation strategy (yes, no)")
		(PR_CREATE_TESTS, value<bool>()->default_value(false), "create test cases")
		(PR_TEST_NAME, value<string>()->default_value("test/examples/example.json"), "test case file name (JSON)")
		(PR_CACHE_SIZE, value<int>()->default_value(DEFAULT_CACHE_SIZE), "cache size (in MB)")
		(PR_EPOCH, value<fvalue>()->default_value(0.5), "epochs number")
		(PR_MARGIN, value<fvalue>()->default_value(0.1), "margin")
		(PR_INPUT, value<string>(), "input file");
}

Configuration GetConfig(vector<string> args){
	positional_options_description opt;
	opt.add(PR_KEY_INPUT, -1);

	variables_map vars;
	store(command_line_parser(args).options(inputOptions).positional(opt).run(), vars);
	notify(vars);

    
	ParametersParser parser(vars);
	Configuration conf = parser.getConfiguration();
	return conf;

}