#include "osvm_test.h"

// TODO: #3 add time 

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

// read ground truth classifier output from json file
pt::ptree test_json_read(string filename){
  pt::ptree root;
  pt::read_json(filename,root);
  return root;
}

// https://www.boost.org/doc/libs/1_73_0/libs/test/doc/html/boost_test/runtime_config/custom_command_line_arguments.html
BOOST_AUTO_TEST_CASE(my_test) {
    string path = "test/examples/";
    string filename = path + "example.json";

    // parse json file
	BOOST_TEST_MESSAGE( "Parsing JSON file :" << filename );
	pt::ptree root;
    pt::read_json(filename,root);

	// get configuration for ground truth
	pt::ptree config = root.get_child("config");
	pt::ptree model = root.get_child("classifier");

	// create command line arguments
	vector<string> arguments;
	BOOST_FOREACH(const pt::ptree::value_type &v, config){
		const string option(v.first.data());
		string value = v.second.data();
		if(option.back() == 'I'){
			value = v.second.data();
		}
		arguments.push_back((string)"-"+option.back());
		arguments.push_back(value);
	}
   	for(int i=0; i < arguments.size(); i++)
      cout << arguments[i] << ' ';
	cout << "\n";

	//BOOST_TEST( test_json_read( filename) == root );
	pt::ptree model_tree;

	initOptions();
	// run program
	positional_options_description opt;
	opt.add(PR_KEY_INPUT, -1);
	
	variables_map vars;
	store(command_line_parser(arguments).options(inputOptions).positional(opt).run(), vars);
	notify(vars);

	if (!vars.count(PR_KEY_HELP)) {
		ParametersParser parser(vars);
		Configuration conf = parser.getConfiguration();

		ApplicationLauncher launcher(conf);
		model_tree = launcher.run();
		pt::write_json("testing.json", model_tree.get_child("classifier"));	

		BOOST_TEST(model == model_tree.get_child("classifier"));
	} else {
		cerr << descr;
	}
	
    // for (const auto & entry : fs::directory_iterator(path))
    //     cout << entry.path() << endl;
}