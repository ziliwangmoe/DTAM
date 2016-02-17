cmake_minimum_required(VERSION 2.8)
project( opencv_cuda )
find_package( OpenCV COMPONENTS core imgproc highgui calib3d cudaimgproc REQUIRED )
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)

macro (add_sources)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND DTAM_SOURCES "${_relPath}/${_src}")
        else()
            list (APPEND DTAM_SOURCES "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (DTAM_SOURCES ${DTAM_SOURCES} PARENT_SCOPE)
    endif()
endmacro(add_sources)

add_subdirectory(drawBall)
add_subdirectory(utils)
add_subdirectory(chamo)

set(OPENGL_3TH_INCLUDE "D:/glew-1.13.0-win32/glew-1.13.0/include;" 
				   "D:/glfw-3.1.2.bin.WIN64/include/GLFW;"
				   "D:/glm-0.9.7.2/glm")
				   
set(OPENGL_3TH_LIB "D:/glew-1.13.0-win32/glew-1.13.0/lib/Release/x64/glew32.lib;" 
				   "D:/glfw-3.1.2.bin.WIN64/lib-vc2013/glfw3dll.lib;")
message(STATUS ${OPENGL_LIBRARIES})


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /wd4819")

set(CUDA_NVCC_FLAGS_DEBUG "-g -G -Xcompiler \"/wd4819\"")
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-gencode arch=compute_30,code=sm_30;--use_fast_math ;--restrict; -lineinfo)
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OPENGL_3TH_INCLUDE})
CUDA_ADD_EXECUTABLE(chamo testprog.cpp ${DTAM_SOURCES} convertAhandaPovRayToStandard.h convertAhandaPovRayToStandard.cpp)
target_link_libraries( chamo ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${OPENGL_3TH_LIB})
#file(COPY loli.jpg DESTINATION ${CMAKE_BINARY_DIR})