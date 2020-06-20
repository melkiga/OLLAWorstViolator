cmake_minimum_required(VERSION 3.13...3.16 FATAL_ERROR)

# project settings
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ".")
set(PROJECT_UNIT_TESTS_NAME ${PROJECT_NAME}_unit_tests)

# include paths
file(GLOB_RECURSE TST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/*.cc )
file(GLOB_RECURSE TST_HDR_FILES ${PROJECT_SOURCE_DIR}/test/*.h )
file(GLOB_RECURSE OSVM_SRC_FILES ${PROJECT_SOURCE_DIR}/src/*/*.cc )
file(GLOB_RECURSE OSVM_HDR_FILES ${PROJECT_SOURCE_DIR}/src/*/*.h )

set(OSVM_SOURCES ${OSVM_SRC_FILES} 
                 ${OSVM_HDR_FILES}
                 ${PROJECT_SOURCE_DIR}/src/launcher.h
                 ${PROJECT_SOURCE_DIR}/src/launcher.cc
                 ${PROJECT_SOURCE_DIR}/src/configuration.h
                 ${PROJECT_SOURCE_DIR}/src/configuration.cc )

# dependencies
find_package(Boost COMPONENTS program_options unit_test_framework REQUIRED)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(GSL REQUIRED)

# include dirs
set(UNIT_TESTS_INCLUDES ${Boost_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS})

# create executable
add_executable(${PROJECT_UNIT_TESTS_NAME} ${TST_SRC_FILES} ${TST_HDR_FILES} ${OSVM_SOURCES})

# set target properties
set_target_properties(${PROJECT_UNIT_TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test/bin/${CMAKE_BUILD_TYPE})

# include dirs + libraries
target_include_directories(${PROJECT_UNIT_TESTS_NAME} PRIVATE ${UNIT_TESTS_INCLUDES})
target_link_libraries(${PROJECT_UNIT_TESTS_NAME} ${Boost_LIBRARIES} ${GSL_LIBRARIES})

enable_testing()
add_test(NAME application_tester COMMAND ${PROJECT_UNIT_TESTS_NAME})
#####################################################




