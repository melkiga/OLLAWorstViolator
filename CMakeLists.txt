#cmake --help-command-list
#cmake --help-variable-list
#cmake --help-property-list

cmake_minimum_required(VERSION 3.0.0)
project(osvm VERSION 0.0.1 LANGUAGES CXX)
MESSAGE("Project name: " ${PROJECT_NAME} ", Version: " ${PROJECT_VERSION})
MESSAGE("Project source directory: " ${PROJECT_SOURCE_DIR})

# get all source files & add them to executable
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE HDR_FILES ${PROJECT_SOURCE_DIR}/src/*.h)
MESSAGE("Adding source files: " ${SRC_FILES} ${HDR_FILES})

# add source files to executable
add_executable(osvm ${SRC_FILES} ${HDR_FILES})

# tar
target_include_directories(osvm
    PUBLIC 
        #$<INSTALL_INTERFACE:include>    
        #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# BOOST
find_package(Boost COMPONENTS program_options REQUIRED)
if (Boost_FOUND)
    MESSAGE(STATUS "  [Found Boost library]")
    MESSAGE(STATUS "    include dir:" ${Boost_INCLUDE_DIRS})
    FOREACH(item ${Boost_LIBRARIES})
        MESSAGE(STATUS "    library dir: " ${item})
    ENDFOREACH()
    # set some environment variables
    #set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
    #set(Boost_USE_STATIC_RUNTIME OFF)
    # include dirs + libraries
    target_include_directories(osvm PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(osvm ${Boost_LIBRARIES})
endif()


# GSL
find_package(GSL REQUIRED)
if (GSL_FOUND)
    MESSAGE(STATUS "  [Found GSL library]")
    MESSAGE(STATUS "    include dir:" ${GSL_INCLUDE_DIRS})
    FOREACH(item ${GSL_LIBRARIES})
        MESSAGE(STATUS "    library dir: " ${item})
    ENDFOREACH()
    # include dirs + libraries
    target_include_directories(osvm PRIVATE ${GSL_INCLUDE_DIRS})
    target_link_libraries(osvm ${GSL_LIBRARIES})
endif()