#ifndef OSVM_TEST_H_
#define OSVM_TEST_H_

#define BOOST_TEST_MODULE application_tester
#include <boost/test/unit_test.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>

#include <vector>
#include <string>
#include <filesystem>

#include "../src/configuration.h"

using namespace std;

namespace pt = boost::property_tree;
namespace fs = std::filesystem;

bool testJSONRead(string filename);
void initOptions();


#endif