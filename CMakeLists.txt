cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/nvcc.exe")
project(cuda_apap LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)


set(OpenCV_DIR "C:/opencv/opencv/build")
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("C:/opencv/opencv/build/install/include")
include_directories("C:/opencv/opencv/build/install/include/opencv2")

message(STATUS "Opencv include dirs = ${OpenCV_INCLUDE_DIRS}")
#include_directories(C:/opencv/opencv/build/include)
include_directories(include)
set(cuSOLVER_LIB_DIR  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/cusolver.lib")

file(GLOB_RECURSE DIR_SRCS    ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
file(GLOB_RECURSE BATCH_SVD "src/test_batch_svd.cu")
file(GLOB_RECURSE DIR_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
list(REMOVE_ITEM DIR_SRCS ${TEST_NODE_SRCS} ${BATCH_SVD})

# ###################################################################################

CUDA_ADD_EXECUTABLE(cuda_apap 
					${DIR_SRCS}
					${DIR_HEADERS}
				)
				
target_compile_features(cuda_apap PUBLIC cxx_std_11)
set_target_properties( cuda_apap
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuda_apap
		${OpenCV_LIBRARIES}
        ${cuSOLVER_LIB_DIR}
)


# ######################################################################################
CUDA_ADD_EXECUTABLE(test_batch_svd
					${BATCH_SVD} src/utils.cu
				)
				
target_compile_features(test_batch_svd PUBLIC cxx_std_11)
set_target_properties( test_batch_svd
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(test_batch_svd
		${OpenCV_LIBRARIES}
        ${cuSOLVER_LIB_DIR}
)


