#ifndef TEST_HELPERS_H_
#define TEST_HELPERS_H_

#include <stdarg.h>
#include <filesystem>

#include "../src/configuration.h"

#define MAX_SIZE 255

typedef const char* LPCSTR;

namespace pt = boost::property_tree;
namespace fs = std::filesystem;

Configuration GetConfig(vector<string> args);
void initOptions();


#endif