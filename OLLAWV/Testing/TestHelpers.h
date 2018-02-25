#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <iostream>


#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/format.hpp>

using namespace boost;
using namespace boost::program_options;

#define MAX_SIZE 255


// ------------------------------------------------------------------------------------------------------------------------------
char* CreateArgBuffer()
{
  char* res = new char[MAX_SIZE + 1];
  memset(res, 0, MAX_SIZE + 1);

  return res;
}

// ------------------------------------------------------------------------------------------------------------------------------
 char** ParseCommand(const char* cmdLine, int& count)
{
  char** res = (char**)calloc(MAX_SIZE, sizeof(char*));

  size_t bufferSize = 0;
  size_t len = strnlen_s(cmdLine, MAX_SIZE);

  // The first argument is the path of the application.  For our test code,
  // this won't matter.
  char* buffer = CreateArgBuffer();
  strcpy_s(buffer, MAX_SIZE, "OLLAWVTester.app");
  res[0] = buffer;
  ++count;

  // Now we can just scan out the input chars from the 'command line'
  buffer = CreateArgBuffer();
  for (size_t i = 0; i < len; i++)
  {
    char next = cmdLine[i];
    if (next == ' ')
    {
      // Thow it on the stack.
      res[count] = buffer;
      ++count;

      // Make a new buffer!
      buffer = CreateArgBuffer();
      bufferSize = 0;
    }
    else
    {
      // We will just reject quotes in our parsing machine.
      if (next != '\"')
      {
        buffer[bufferSize] = next;
        ++bufferSize;
      }
    }
  }

  // Commit the rest of the buffer data.
  if (bufferSize > 0)
  {
    res[count] = buffer;
    ++count;
  }

  return res;
}


// ------------------------------------------------------------------------------------------------------------------------------
// This cleans up the memory from the output of ParseCommand.
void CleanupCommandArgs(const char** args, int count)
{
}

#endif
