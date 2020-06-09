#define BOOST_TEST_MODULE application_tester
#include <boost/test/unit_test.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

using namespace std;

namespace pt = boost::property_tree;
namespace fs = std::filesystem;