cmake_minimum_required(VERSION 3.13...3.16 FATAL_ERROR)
project(osvm VERSION 0.0.1 LANGUAGES CXX)
message("Project Name: " ${PROJECT_NAME} ", Version: " ${PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 17)

# set variables
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src") # full path to src
set(INSTALL_BIN_DIR "${PROJECT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}")
set(INSTALL_LIB_DIR "${PROJECT_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE}")

# get all source files & add them to executable
message("Finding source and header files...")
file(GLOB_RECURSE SRC_FILES ${SOURCE_DIR}/*.cc )
file(GLOB_RECURSE HDR_FILES ${SOURCE_DIR}/*.h )

# create executable
message("Creating osvm executable...")
add_executable(osvm ${SRC_FILES} ${SOURCE_DIR}/svm/classify.h ${SOURCE_DIR}/svm/strategy.h)

# add include directories
target_include_directories(osvm
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

set_target_properties(osvm
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${INSTALL_LIB_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${INSTALL_LIB_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${INSTALL_BIN_DIR}"
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
find_package(Boost COMPONENTS program_options unit_test_framework REQUIRED)
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

#####################################################
# build test 
ENABLE_TESTING()

# creates the executable
file(GLOB_RECURSE TST_SRC_FILES ${CMAKE_SOURCE_DIR}/test/*.cc )
add_executable(tests ${TST_SRC_FILES})
set_target_properties(tests
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${INSTALL_LIB_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${INSTALL_LIB_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${INSTALL_BIN_DIR}"
)
# indicates the include paths
target_include_directories(tests PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(tests ${Boost_LIBRARIES})
target_include_directories(tests PRIVATE ${GSL_INCLUDE_DIRS})
target_link_libraries(tests ${GSL_LIBRARIES})
target_include_directories(tests PRIVATE ${SOURCE_DIR})

# declares a test with our executable
add_test(NAME application_tester COMMAND tests)