#ifndef OSVM_TEST_H_
#define OSVM_TEST_H_

#define BOOST_TEST_MODULE application_tester
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>

#include <stdarg.h>

#include "../src/configuration.h"
#include "../src/launcher.h"

#define MAX_SIZE 255
#define TEST_EXAMPLE_PATH "test/examples/"

typedef const char* LPCSTR;

namespace pt = boost::property_tree;
namespace bdata = boost::unit_test::data;

Configuration GetConfig(vector<string> args);
void initOptions(vector<string> &arguments, bopt::variables_map& vars);

void parse_test_config(vector<string> &arguments, const pt::ptree& config);

ostream& operator<<(ostream& os, pt::ptree tree);

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}


#endif