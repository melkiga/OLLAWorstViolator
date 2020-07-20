#include "osvm_test.h"

const char* test_example_filenames[] = {"teach_test.json", "iris_test.json"};

// TODO: #3 add time 

// Get the options ready.
void initOptions(vector<string> &arguments, bopt::variables_map& vars){
	// program options (command line args)
	string usage = (format("Usage: %s [OPTION]... [FILE]\n") % PACKAGE).str();
	string descr = "Perform SVM training for the given data set [FILE].\n";
	string options = "Available options";
	bopt::options_description inputOptions(usage + descr + options);
	inputOptions.add_options()
		(PR_HELP, "produce help message")
		(PR_C_LOW, bopt::value<fvalue>()->default_value(0.001), "C value (lower bound)")
		(PR_C_HIGH, bopt::value<fvalue>()->default_value(10000.0), "C value (upper bound)")
		(PR_G_LOW, bopt::value<fvalue>()->default_value(0.0009765625), "gamma value (lower bound)")
		(PR_G_HIGH, bopt::value<fvalue>()->default_value(16.0), "gamma value (upper bound)")
		(PR_RES, bopt::value<int>()->default_value(8), "resolution (for C and gamma)")
		(PR_OUTER_FLD, bopt::value<int>()->default_value(1), "outer folds")
		(PR_INNER_FLD, bopt::value<int>()->default_value(10), "inner folds")
		(PR_BIAS_CALCULATION, bopt::value<string>()->default_value(BIAS_CALCULATION_YES), "bias evaluation strategy (yes, no)")
		(PR_CREATE_TESTS, bopt::value<bool>()->default_value(false), "create test cases")
		(PR_TEST_NAME, bopt::value<string>()->default_value("test/examples/example.json"), "test case file name (JSON)")
		(PR_CACHE_SIZE, bopt::value<int>()->default_value(DEFAULT_CACHE_SIZE), "cache size (in MB)")
		(PR_EPOCH, bopt::value<fvalue>()->default_value(0.5), "epochs number")
		(PR_MARGIN, bopt::value<fvalue>()->default_value(0.1), "margin")
		(PR_INPUT, bopt::value<string>(), "input file");

	bopt::positional_options_description opt;
	opt.add(PR_KEY_INPUT, -1);
	bopt::store(bopt::command_line_parser(arguments).options(inputOptions).positional(opt).run(), vars);
	bopt::notify(vars);
}

void parse_test_config(vector<string> &arguments, const pt::ptree& config){
	
	BOOST_FOREACH(const pt::ptree::value_type &v, config){
		const string option(v.first.data());
		string value = v.second.data();
		// get dataset
		if(option.back() == 'I'){
			value = v.second.data();
		}
		arguments.push_back((string)"-"+option.back());
		arguments.push_back(value);
	}
}

BOOST_DATA_TEST_CASE(test_one, bdata::make(test_example_filenames), array_element){
	cout << "Running test one: "
    << array_element
    << endl;

	// get path to test example output
	string example_full_path = TEST_EXAMPLE_PATH + string(array_element);
	
	// parse example json file
	BOOST_TEST_MESSAGE( "Parsing test example JSON file :" << array_element );
	pt::ptree root;
	pt::read_json(example_full_path,root);
	
	// get test run configuration and resulting classifier
	pt::ptree config = root.get_child("config");
	pt::ptree model = root.get_child("classifier");
	
	// parsing test config from json
	vector<string> arguments;
	parse_test_config(arguments, config);
	cout << "Test example arguments: " << arguments << endl;
	BOOST_TEST_MESSAGE("Test Example Model :");
	pt::write_json(cout,model);
	
	// start run setup
	bopt::variables_map vars;
	initOptions(arguments, vars);
	ParametersParser parser(vars);
	Configuration conf = parser.getConfiguration();

	// run application
	ApplicationLauncher launcher(conf);
	pt::ptree model_tree;
	model_tree.put_child("config", pt::ptree());
	model_tree.put_child("classifier", pt::ptree());
	launcher.run(model_tree);
	pt::write_json(cout,model_tree.get_child("classifier"));
	
	// test output with test example
	BOOST_TEST(model == model_tree.get_child("classifier"));
}

BOOST_AUTO_TEST_CASE( test_multiple_runs )
{
  	// get path to test example output
	string example_full_path = TEST_EXAMPLE_PATH + string("iris_test.json");
	
	// parse example json file
	pt::ptree root;
	pt::read_json(example_full_path,root);
	
	// get test run configuration and resulting classifier
	pt::ptree config = root.get_child("config");
	pt::ptree model = root.get_child("classifier");
	
	// parsing test config from json
	vector<string> arguments;
	parse_test_config(arguments, config);
	cout << "Test example arguments: " << arguments << endl;

	// start run setup
	bopt::variables_map vars;
	initOptions(arguments, vars);
	ParametersParser parser(vars);
	Configuration conf = parser.getConfiguration();

	// run application
	ApplicationLauncher launcher(conf);
	pt::ptree model_tree;
	model_tree.put_child("config", pt::ptree());
	model_tree.put_child("classifier", pt::ptree());
	launcher.run(model_tree);
	pt::write_json(cout,model_tree.get_child("classifier"));

	// second run
	pt::ptree model_tree_two;
	model_tree_two.put_child("config", pt::ptree());
	model_tree_two.put_child("classifier", pt::ptree());
	launcher.run(model_tree_two);
	pt::write_json(cout,model_tree_two.get_child("classifier"));

	// test output with test example
	BOOST_TEST(model_tree.get_child("classifier") == model_tree_two.get_child("classifier"));
}