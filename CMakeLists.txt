cmake_minimum_required(VERSION 3.13...3.16 FATAL_ERROR)
project(osvm VERSION 0.0.1 LANGUAGES CXX)
message("Project Name: " ${PROJECT_NAME} ", Version: " ${PROJECT_VERSION})

# get all source files & add them to executable
message("Finding source and header files...")
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cc )
file(GLOB_RECURSE HDR_FILES ${PROJECT_SOURCE_DIR}/src/*.h )

# create executable
message("Creating osvm executable...")
add_executable(osvm ${SRC_FILES} ${HDR_FILES})

# add include directories
target_include_directories(osvm
    PUBLIC 
        #$<INSTALL_INTERFACE:include>    
        #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(osvm PRIVATE "VERBOSITY=$<IF:$<CONFIG:Debug>,30,10>")

set_target_properties(osvm
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/lib/Debug"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/lib/Debug"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin/Debug"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/lib/Release"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/lib/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin/Release"
)

##############################################
# Create debug and release target paths
add_custom_target(debug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target osvm
    COMMENT "Creating the executable in the debug mode.")

add_custom_target(release
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR} 
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target osvm
    COMMENT "Creating the executable in the release mode.")

##############################################
# Declare dependencies
# BOOST
message("Searching for Boost...")
find_package(Boost COMPONENTS program_options REQUIRED)
if (NOT Boost_FOUND)
    message(FATAL_ERROR "NOTICE: This demo requires Boost and will not be compiled.")
endif()

# set some environment variables
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
# include dirs + libraries
target_include_directories(osvm PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(osvm ${Boost_LIBRARIES})

# GSL
message("Searching for GSL...")
find_package(GSL REQUIRED)
if (NOT GSL_FOUND)
    message(FATAL_ERROR "NOTICE: This demo requires GSL and will not be compiled.")
endif()
# include dirs + libraries
target_include_directories(osvm PRIVATE ${GSL_INCLUDE_DIRS})
target_link_libraries(osvm ${GSL_LIBRARIES})

##############################################
# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

file(GLOB_RECURSE TST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/*.cc )
file(GLOB_RECURSE TST_HDR_FILES ${PROJECT_SOURCE_DIR}/tests/*.h )

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(test tests/test.cc)
#add_executable(test ${SRC_FILES} ${HDR_FILES} ${TST_SRC_FILES} ${TST_HDR_FILES})

set_target_properties(test
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY_TEST "${CMAKE_SOURCE_DIR}/lib/Test"
    LIBRARY_OUTPUT_DIRECTORY_TEST "${CMAKE_SOURCE_DIR}/lib/Test"
    RUNTIME_OUTPUT_DIRECTORY_TEST "${CMAKE_SOURCE_DIR}/bin/Test"
)

target_link_libraries(test gtest)
add_test(NAME testing COMMAND test)