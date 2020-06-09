#include "osvm_test.h"

bool test_json_read(string filename){
  pt::ptree root;
  pt::ptree rootest;
  pt::read_json(filename,root);
  pt::read_json(filename,rootest);
  //pt::write_json(std::cout, root);
  return (root==rootest)? true : false ;
}

BOOST_AUTO_TEST_CASE(my_test) {
    string filename = "test/examples/example.json";
    BOOST_CHECK_EQUAL( test_json_read( filename ), true );

    // string path = "test/examples/"; //TODO: this should be a cmdline arg
    // for (const auto & entry : fs::directory_iterator(path))
    //     cout << entry.path() << endl;
    // test_json_read(filename);
}