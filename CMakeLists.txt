cmake_minimum_required(VERSION 2.8)

project(jusha)
set (jusha_VERSION 0.1.0)
set (jusha_VERSION_MAJOR 0)
set (jusha_VERSION_MINOR 1)
set (jusha_VERSION_PATCH 0)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")

# CUDA is always required.
find_package(CUDA REQUIRED)
#set(CUDA_PROPAGATE_HOST_FLAGS off)


# Possible build types.
#if(NOT CMAKE_BUILD_TYPE)
#    set(CMAKE_BUILD_TYPE Debug CACHE STRING
#        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
#    FORCE)
#endif()

#set(CUDA_SEPARABLE_COMPILATION ON)
set(BUILD_SHARED_LIBS OFF)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic -Wall -Werror")
set(CUDA_PROPAGATE_HOST_FLAGS off)


set(CUDA_BLOCKSIZE 256 CACHE STRING "CUDA block size.")

# CUDA architecture options.
set(GENCODE_SM30 "-gencode=arch=compute_30,code=sm_30")
set(GENCODE_SM35 "-gencode=arch=compute_35,code=sm_35")
set(GENCODE_SM20 "-gencode=arch=compute_20,code=sm_20")

# Base CUDA nvcc flags taken from the original makefiles.
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${GENCODE_SM20} ${GENCODE_SM30} ${GENCODE_SM35})


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0 -DDEBUG")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG -O0")
else()
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
endif()


# Make the gampack headers visible without extra paths.
include_directories(src)
include_directories("${CMAKE_BINARY_DIR}/src")
include_directories("${CMAKE_BINARY_DIR}/src/lib/cub-1.3.2")

# this flag is for gcc only

if (APPLE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set (CUDA_HOST_COMPILER g++)
#set (CMAKE_CXX_COMPILER g++)
elseif (UNIX)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()
#-std=c++0x
# Split the library in two parts, the CUDA one and the CPP one.
set(CPP_SRC_LIST
    src/utility.cpp
    src/timer.cpp
    src/cuda/cuda_config.cpp
)

set(CU_SRC_LIST
    src/cuda/utility.cu
    src/cuda/bitmap_scan.cu

#    src/cuda/dummy.cu
)

# Headers are here just so CMake will put them in project files.
set(HEADERS_LIST
    src/timer.h
    src/utility.h
    src/cuda/utility.h
    src/cuda/bitmap_scan.h
)

if(UNIX)
    add_definitions(-DRESTRICT=__restrict__)
endif()


# Must set this because size_t etc is shared between NVCC and Host - only really matters on Windows
set(CUDA_64_BIT_DEVICE_CODE ON)

if(UNIX)
#  add_library(jusha_c SHARED ${CPP_SRC_LIST} ${HEADERS_LIST})
  cuda_add_library(jusha SHARED ${CU_SRC_LIST} ${CPP_SRC_LIST} ${HEADERS_LIST})
#  target_link_library(jusha jusha_c jusha_cu)
else()
#  cuda_add_library(jusha SHARED ${CPP_SRC_LIST} ${CU_SRC_LIST} ${HEADERS_LIST})
  cuda_add_library(jusha STATIC ${CPP_SRC_LIST} ${CU_SRC_LIST} ${HEADERS_LIST})
endif()


#this must be after all set(GAMPACK* "#define HAVE...")
#configure_file("${CMAKE_SOURCE_DIR}/src/jusha_config.h.in.cmake" "${CMAKE_BINARY_DIR}/src/jusha_config.h")

add_subdirectory (tests)
#add_subdirectory (utest)

#add_executable(test_block2 src/test_block2.cpp)
# Make the gampack headers visible without extra paths.
#target_link_libraries (test_block2 gampack)


# for distribution
set (CPACK_PACKAGE_VERSION_MAJOR ${jusha_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR ${jusha_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH ${jusha_VERSION_PATCH})
set (CPACK_SOURCE_TBZ2 "OFF")
set (CPACK_SOURCE_TGZ "ON")
set (CPACK_SOURCE_TZ "OFF")

#set (CPACK_SOURCE_IGNORE_FILES "/dist/;/drop/;/example/;/examples/;cudalt.py;.*~;Makefile.am;.*sln;.*vcxproj*;.*png;.*eps;.*bib")
# set (CPACK_PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/CPackOptions.cmake")

INSTALL(TARGETS jusha jusha  RUNTIME DESTINATION bin  LIBRARY DESTINATION lib  ARCHIVE DESTINATION lib)
#install (FILES src/gampack.h src/gampack_enum.h src/gampack_error.h src/platform_config.h  ${CMAKE_BINARY_DIR}/src/gampack_config.h DESTINATION include)
#install(DIRECTORY src/ DESTINATION include/srt FILES_MATCHING PATTERN "srt_*.h")
#install (FILES src/tinyvec.h src/device_allocator.h src/utility.h src/scan.h src/srt/shared_ptr.h
#         src/srt/srt_thread.h src/srt/srt_this_thread.h DESTINATION include/srt)
#install (FILES src/srt_vector.h src/srt_threadmanager.h src/RmtXchg.h src/RmtXchg_hybrid.h 
#         src/RmtXchg_template.h src/srt_thrust.h src/srt_thrust_cpu.h src/communicator_template.h 
#	 src/communicator.h src/DistriObj.h src/gampack_mpi.h src/srt_exception.h 
#         src/srt_error.h src/gampack_expert.h DESTINATION include)

#install(DIRECTORY src/ DESTINATION src FILES_MATCHING PATTERN "srt_*.cpp")
#install(DIRECTORY src/ DESTINATION src FILES_MATCHING PATTERN "srt_*.cu")
# will rename scan.h to srt_scan.h
#install (FILES src/scan.h DESTINATION include/srt)

include (CPack)
