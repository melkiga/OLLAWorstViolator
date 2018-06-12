#ifndef TEST_HELPERS_H_
#define TEST_HELPERS_H_

#include <exception>
#include <stdarg.h>

#include "../../osvm/config.h"
#include "../../osvm/src/configuration.h"

#define MAX_SIZE 255

// NOTE: This should come from windows.h on windows builds.
typedef const char* LPCSTR;


char** ParseCommand(const char* cmdLine, int& count);
void CleanupCommandArgs( char** args, int count);
Configuration GetConfiguration(char* input);
void InitOptions();
char* CreateArgBuffer();


// A simple way to throw an exception in our test code.
void ThrowTestException(LPCSTR fmt, ...);




// ====================================================================================================
class Check
{
  public:
    template <typename T>
    static void AreEqual(T expected, T actual, LPCSTR message = nullptr)
    {
      bool equal = expected == actual;
      if (!equal)
      {
        // NOTE: specialized templates to get the correct format string would be good too.
        ThrowTestException("%s: Expected %u and got %u", message, expected, actual);
      }
    }
};


#endif
