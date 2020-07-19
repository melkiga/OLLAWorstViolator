cmake_minimum_required(VERSION 3.1...3.16 FATAL_ERROR)
project(osvm VERSION 0.0.1 LANGUAGES CXX)
enable_testing()
message("Project Name: " ${PROJECT_NAME} ", Version: " ${PROJECT_VERSION})

# set install directories
set(INSTALL_BIN_DIR "${PROJECT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}")

# set cmake standard
set(CMAKE_CXX_STANDARD 17)

# include paths
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src") # full path to src

# get all source files & add them to executable
file(GLOB_RECURSE SRC_FILES ${SOURCE_DIR}/*.cc )
file(GLOB_RECURSE HDR_FILES ${SOURCE_DIR}/*.h )

# include tests
add_subdirectory(test)

# create executable
message("Creating osvm executable...")
add_executable(osvm ${SRC_FILES} ${HDR_FILES})

# add include directories
set_target_properties(osvm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${INSTALL_BIN_DIR}")

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
find_package(Boost 1.73 COMPONENTS program_options unit_test_framework REQUIRED)

# set some environment variables
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

# include dirs + libraries
if(Boost_FOUND)
    target_include_directories(osvm PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(osvm ${Boost_LIBRARIES})
endif()

# GSL
find_package(GSL REQUIRED)
if(GSL_FOUND)
    target_link_libraries(osvm GSL::gsl GSL::gslcblas)
endif()


