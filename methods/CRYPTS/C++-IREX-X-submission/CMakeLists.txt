cmake_minimum_required(VERSION 3.1)

project(nullImpl LANGUAGES C CXX)

# set(CMAKE_CXX_CLANG_TIDY "clang-tidy;--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy")

add_library ("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" SHARED src/cryptsimplfrvt1N.cpp src/ortools_linprog.cpp)
target_include_directories("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" PUBLIC include)
target_compile_features("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" PUBLIC cxx_std_17)

set(FRVT_DIR "" CACHE STRING "FRVT Root Directory")
if( FRVT_DIR ) 
    target_include_directories("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" PUBLIC "${FRVT_DIR}/1N/src/include")
    target_include_directories("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" PUBLIC "${FRVT_DIR}/common/src/include")
endif()

#find_package(ortools CONFIG REQUIRED)
#target_include_directories("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" PRIVATE "${ortools_INCLUDE_DIRS}")
#target_link_libraries("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" ortools)

add_subdirectory(or-tools)
target_link_libraries("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" ortools::ortools)

set(OpenCV_CUDA OFF)

find_package(OpenCV COMPONENTS core imgcodecs REQUIRED)
target_include_directories("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" PRIVATE "${OpenCV_INCLUDE_DIRS}")
target_link_libraries("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" "${OpenCV_LIBS}")

find_package(Torch REQUIRED)
target_include_directories("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries("frvt_1N_nd_cvrl_crypts_${FRVT_VER}" "${TORCH_LIBRARIES}")

install(TARGETS "frvt_1N_nd_cvrl_crypts_${FRVT_VER}"
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)
