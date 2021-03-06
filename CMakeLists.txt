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
set(CMAKE_FIND_LIBRARY_SUFFIXES .a .so)
find_package(CUDA REQUIRED)


# openGL stuff
# glfw
if (APPLE)
include_directories(/opt/local/include/)
endif (APPLE)
# GLUT
#find_package(GLUT REQUIRED)
#include_directories(${GLUT_INCLUDE_DIRS})
#link_directories(${GLUT_LIBRARY_DIRS})
#add_definitions(${GLUT_DEFINITIONS})
#if(NOT GLUT_FOUND)
#    message(ERROR " GLUT not found!")
#endif(NOT GLUT_FOUND)

find_library (GLFW_LIB_LIBRARIES  NAMES glfw glfw3
  PATHS /opt/local/lib /usr/local /usr/local/lib
)
message(STATUS "Found glfw library @ ${GLFW_LIB_LIBRARIES}")
set (CUDA_PROPAGATE_HOST_FLAGS OFF)
#########################################################
# FIND OPENGL
#########################################################
find_package(OpenGL REQUIRED)

include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
else ()
	if (UNIX)
	    set(GL_LIBRARIES GLU X11 Xxf86vm Xrandr Xi Xcursor  Xinerama)
	endif()
	if (APPLE)
	    set(GL_LIBRARIES ${OPENGL_LIBRARIES})
	endif()

endif(NOT OPENGL_FOUND)
message(STATUS "Found OpenGL @ ${OPENGL_LIBRARIES}")

#find_package(HDF5 REQUIRED)
#find_library (HDF5HL_LIBRARY  NAMES hdf5_hl
#              PATHS ${HDF5_LIBRARY_DIRS}
#              )
      
# Possible build types.
#if(NOT CMAKE_BUILD_TYPE)
#    set(CMAKE_BUILD_TYPE Debug CACHE STRING
#        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
#    FORCE)
#endif()

#set(CUDA_SEPARABLE_COMPILATION ON)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
#set(CUDA_PROPAGATE_HOST_FLAGS ON)


set(CUDA_BLOCKSIZE 256 CACHE STRING "CUDA block size.")

# CUDA architecture options.
#set(GENCODE_SM35 "-gencode=arch=compute_35,code=sm_35")
set(GENCODE_SM50 "-gencode=arch=compute_50,code=sm_50")
set(GENCODE_SM52 "-gencode=arch=compute_52,code=sm_52")
set(GENCODE_SM30 "-gencode=arch=compute_30,code=sm_30")
set(GENCODE_SM37 "-gencode=arch=compute_37,code=sm_37")
set(GENCODE_SM20 "-gencode=arch=compute_20,code=sm_20")

# Base CUDA nvcc flags taken from the original makefiles.
#if (APPLE)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${GENCODE_SM35})
#else ()
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${GENCODE_SM20} ${GENCODE_SM30} ${GENCODE_SM35} ${GENCODE_SM50})
#endif()
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${GENCODE_SM35} ${GENCODE_SM50})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} )

#
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#  set(CUDA_NVCC_FLAGS "-O0" "-g")
#  add_definitions("-DDEBUG")	
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG -O0 -g")
else()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
endif()


# Make the gampack headers visible without extra paths.
include_directories(src)
include_directories("${CMAKE_BINARY_DIR}/src")
include_directories("${CMAKE_SOURCE_DIR}/src/lib/cub-1.3.2")
include_directories("${CMAKE_SOURCE_DIR}/src/lib/catch")
#include_directories("${HDF5_INCLUDE_DIRS}")
#include_directories("${HEMI_ROOT})

# this flag is for gcc only
set (CMAKE_MACOSX_RPATH On)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#if (APPLE)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif (UNIX)

#set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-XCompiler, \\"-std=C++11\\"")
#endif()
#-std=c++0x
# Split the library in two parts, the CUDA one and the CPP one.
set(CPP_SRC_LIST
    src/utility.cpp
    src/timer.cpp
    src/cuda/cuda_config.cpp
    src/cuda/heap_manager.cpp
    src/cuda/heap_allocator.cpp
    src/cuda/kernel.cpp

    src/sparse/solver/diag_precond.cpp
    src/sparse/solver/jacobi_precond.cpp    
	src/sparse/distri_obj.cpp    
    src/sparse/matrix_reader.cpp
    src/sparse/mmio.cpp
    src/sparse/csr_matrix.cpp
    

	src/sparse/parser/su2_mesh_parser.cpp

    
)

set(CU_SRC_LIST
    src/cuda/utility.cu
    src/cuda/bitmap_scan.cu
    src/cuda/array.cu    

    src/sparse/csr_matrix.cu    
	src/sparse/mesh.cu


#    src/cuda/dummy.cu
#if(GLFW_LIB_LIBRARIES)
#	src/sparse/view/2d_viewer.cu
#    src/sparse/view/opengl_helper.cpp
#endif
)

# Headers are here just so CMake will put them in project files.
set(HEADERS_LIST
    src/timer.h
    src/utility.h

    src/cuda/bitmap_scan.h
    src/cuda/heap_manager.h
    src/cuda/heap_allocator.h
    src/cuda/array.h
    src/cuda/array_util.h
    src/cuda/allocator.h
	src/cuda/for_each.hpp
	src/cuda/kernel.hpp
	src/cuda/intrinsic.hpp
	src/cuda/policy.hpp

    src/sparse/matrix.h
	src/sparse/distri_obj.h    
    src/sparse/csr_matrix.h
    src/sparse/matrix_reader.h
    src/sparse/mmio.h
	src/sparse/mesh.h

	src/sparse/solver/precond.h
    
	src/sparse/parser/su2_mesh_parser.h
#if(GLFW_LIB_LIBRARIES)
#	src/sparse/view/2d_viewer.h
#	src/sparse/view/opengl_helper.h
#endif

)

# Must set this because size_t etc is shared between NVCC and Host - only really matters on Windows
set(CUDA_64_BIT_DEVICE_CODE ON)

#if(UNIX)
#  add_library(jusha_c SHARED ${CPP_SRC_LIST} ${HEADERS_LIST})
  cuda_add_library(jusha STATIC ${CU_SRC_LIST} ${CPP_SRC_LIST} ${HEADERS_LIST}
	OPTIONS ${GENCODE_SM30} ${GENCODE_SM37} ${GENCODE_SM52} "-std=c++11" # "--relaxed-constexpr" # "--resource-usage" "-v"
	RELEASE --use_fast_math;-g;-O3; ${GENCODE_SM37} ${GENCODE_SM30}  ${GENCODE_SM52} 
	DEBUG -g;-O0;-DDEBUG; ${GENCODE_SM37} ${GENCODE_SM30}  ${GENCODE_SM52} 
    ) # "-Xcompiler ,\"-std=c++11\"")
#  target_link_library(jusha jusha_c jusha_cu)
#else()
#  cuda_add_library(jusha SHARED ${CPP_SRC_LIST} ${CU_SRC_LIST} ${HEADERS_LIST})
#  cuda_add_library(jusha STATIC ${CPP_SRC_LIST} ${CU_SRC_LIST} ${HEADERS_LIST})
#endif()

#if (APPLE)
#target_link_libraries(jusha /usr/local/cuda/lib/libcudadevrt.a)
#elseif (UNIX)
#target_link_libraries(jusha /usr/local/cuda/lib64/libcudadevrt.a)
#endif()

target_link_libraries(jusha ${CUDA_curand_LIBRARY})
#include_directories(/opt/local/lib)
if (GLFW_LIB_LIBRARIES)
target_link_libraries(jusha ${GLFW_LIB_LIBRARIES})
endif()
#
#      message(STATUS, "hdf5 lib names ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES}")
#      message(STATUS "hdf5 high level ${HDF5HL_LIBRARY}")
#target_link_libraries(jusha ${OPENGL_LIBRARIES}) #  ${HDF5_LIBRARIES} ${HDF5HL_LIBRARY})
target_link_libraries(jusha ${GL_LIBRARIES}) #  ${HDF5_LIBRARIES} ${HDF5HL_LIBRARY})
#configure_file("${CMAKE_SOURCE_DIR}/src/jusha_config.h.in.cmake" "${CMAKE_BINARY_DIR}/src/jusha_config.h")

add_subdirectory (tests)
#add_subdirectory (utest)



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
INSTALL(DIRECTORY src/ DESTINATION include/jusha FILES_MATCHING PATTERN "*.h")
INSTALL(DIRECTORY src/ DESTINATION include/jusha FILES_MATCHING PATTERN "*.hpp")
include (CPack)
