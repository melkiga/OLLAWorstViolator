#include "osvm_test.h"

// read ground truth classifier output from json file
pt::ptree test_json_read(string filename){
  pt::ptree root;
  pt::read_json(filename,root);
  return root;
}

BOOST_AUTO_TEST_CASE(my_test) {
    string path = "test/examples/"; //TODO: this should be a cmdline arg
    string filename = path + "example.json";

    // parse json file
	pt::ptree root;
    pt::read_json(filename,root);

	// get configuration for ground truth
	pt::ptree config = root.get_child("config");
	pt::ptree model = root.get_child("classifier");
	pt::write_json(cout,config);

	// create command line arguments
	vector<string> arguments;
	BOOST_FOREACH(const pt::ptree::value_type &v, config){
		const string option(v.first.data());
		string value = v.second.data();
		if(option.back() == 'I'){
			value = "../" + v.second.data();
		}
		arguments.push_back((string)"-"+option.back());
		arguments.push_back(value);
	}
   	for(int i=0; i < arguments.size(); i++)
      cout << arguments[i] << ' ';
	cout << "\n";
	// // get argument string
	// string filen = config.get<string>(PR_TEST_NAME);
	// cout << filen << "\n";
	initOptions();

	// run program
	Configuration conf = GetConfig(arguments);
	
	BOOST_TEST( test_json_read( filename) == root );

    for (const auto & entry : fs::directory_iterator(path))
        cout << entry.path() << endl;
}