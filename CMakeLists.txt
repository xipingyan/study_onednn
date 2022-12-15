#===============================================================================
# Copyright 2016-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

cmake_minimum_required(VERSION 3.2)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert RelWithMDd...")
endif()

set(PROJECT_NAME "study_onednn")
set(PROJECT_FULL_NAME "Study oneAPI Deep Neural Network Library (oneDNN)")
set(PROJECT_VERSION "2.7.0")

add_subdirectory(3rd/oneDNN)

if (CMAKE_VERSION VERSION_LESS 3.0)
    project(${PROJECT_NAME} C CXX)
else()
    cmake_policy(SET CMP0048 NEW)
    project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)
endif()

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "oneDNN supports 64 bit platforms only")
endif()

# Set the target architecture.
if(NOT DNNL_TARGET_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
        set(DNNL_TARGET_ARCH "AARCH64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64.*|PPC64.*|powerpc64.*)")
        set(DNNL_TARGET_ARCH "PPC64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(s390x.*|S390X.*)")
        set(DNNL_TARGET_ARCH "S390X")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(rv.*|RV.*|riscv.*|RISCV.*)")
        set(DNNL_TARGET_ARCH "RV64")
    else()
        set(DNNL_TARGET_ARCH "X64")
    endif()
endif()
message(STATUS "DNNL_TARGET_ARCH: ${DNNL_TARGET_ARCH}")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_SRC_CCXX_FLAGS)       # SRC specifics
set(CMAKE_EXAMPLE_CCXX_FLAGS)   # EXAMPLE specifics
set(CMAKE_TEST_CCXX_FLAGS)      # TESTS specifics

string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)

if(UNIX OR MINGW)
    if(CMAKE_VERSION VERSION_LESS "3.1.0")
        # No CMAKE_<lang>_STANDARD, so add directly to CMAKE_<lang>_FLAGS
        # (prepended so the user can override)
        set(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")
        # Let SYCL to choose the C++ standard it needs.
        if(NOT DNNL_WITH_SYCL)
            set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
        endif()
    else()
        # CMAKE_<lang>_STANDARD support, so set it to our defaults, unless
        # overridden by the user
        if(NOT DEFINED CMAKE_C_STANDARD)
            set(CMAKE_C_STANDARD 99)
        endif()
        if(NOT DEFINED CMAKE_CXX_STANDARD AND NOT DNNL_WITH_SYCL)
            set(CMAKE_CXX_STANDARD 11)
        endif()

        # Disable -std=gnuXX and -std=gnu++XX
        set(CMAKE_C_EXTENSIONS OFF)
        set(CMAKE_CXX_EXTENSIONS OFF)
    endif()
endif()

# one dnn


# find_package(dnnl REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/3rd/oneDNN/build/install/lib/cmake/dnnl/)
# add_subdirectory(${DNNL_DIR} DNNL)
include_directories(${DNNL_DIR}/include)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/oneDNN/include/
    /home/xiping/mydisk2_2T/ov_cpu/study_onednn/3rd/oneDNN/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/header/
    )

AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC_FILES)

FILE(GLOB_RECURSE HEADER_FILES ${CMAKE_SOURCE_DIR}/header/*.hpp)
SOURCE_GROUP("Include" FILES ${HEADER_FILES})

ADD_EXECUTABLE(${PROJECT_NAME} ${HEADER_FILES} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} dnnl)
# target_link_libraries(${PROJECT_NAME} DNNL::dnnl)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
#     $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)
