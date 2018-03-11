#ifndef TEST_HELPERS_H_
#define TEST_HELPERS_H_

#include "../../osvm/config.h"
#include "../../osvm/src/configuration.h"

#define MAX_SIZE 255

char** ParseCommand(const char* cmdLine, int& count);
void CleanupCommandArgs( char** args, int count);
Configuration GetConfiguration(char* input);
void InitOptions();
char* CreateArgBuffer();

#endif
