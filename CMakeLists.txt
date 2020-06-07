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
# enable testing
enable_testing()

# get all source files & add them to executable
message("Finding source and header files...")
file(GLOB_RECURSE TST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/*.cc )
file(GLOB_RECURSE TST_HDR_FILES ${PROJECT_SOURCE_DIR}/test/*.h )

# create executable
message("Creating osvm test executable...")
add_executable(osvm_tests ${TST_SRC_FILES} ${TST_HDR_FILES})

