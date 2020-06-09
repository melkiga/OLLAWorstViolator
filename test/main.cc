#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#define BOOST_TEST_MODULE application_tester
#include <boost/test/unit_test.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>

using namespace std;
namespace pt = boost::property_tree;

int add( int i, int j ) { return i+j; }

void test_json_read(string filename){
  pt::ptree root;
  pt::read_json(filename,root);
  pt::write_json(std::cout, root);
}

BOOST_AUTO_TEST_CASE(my_test) {
    // seven ways to detect and report the same error:
    BOOST_CHECK( add( 2,2 ) == 4 );        // #1 continues on error

    BOOST_REQUIRE( add( 2,2 ) == 4 );      // #2 throws on error

    if( add( 2,2 ) != 4 )
      BOOST_ERROR( "Ouch..." );            // #3 continues on error

    if( add( 2,2 ) != 4 )
      BOOST_FAIL( "Ouch..." );             // #4 throws on error

    if( add( 2,2 ) != 4 ) throw "Ouch..."; // #5 throws on error

    BOOST_CHECK_MESSAGE( add( 2,2 ) == 4,  // #6 continues on error
                         "add(..) result: " << add( 2,2 ) );

    BOOST_CHECK_EQUAL( add( 2,2 ), 4 );	  // #7 continues on error

    string filename = "test/examples/example.json";
    test_json_read(filename);
}