#ifndef OSVM_TEST_H_
#define OSVM_TEST_H_

#define BOOST_TEST_MODULE application_tester
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>

#include "test_helpers.h"



ostream& operator<<(ostream& os, pt::ptree tree);
pt::ptree test_json_read(string filename);

#endif