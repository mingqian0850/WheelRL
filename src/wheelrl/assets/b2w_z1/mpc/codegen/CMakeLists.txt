cmake_minimum_required(VERSION 3.10)
project(CompiledSolver)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

find_library(FATROP_LIB fatrop REQUIRED)
find_library(BLASFEO_LIB blasfeo REQUIRED)

if(DEFINED ENV{CONDA_PREFIX})
  include_directories($ENV{CONDA_PREFIX}/include)
else()
  include_directories(/opt/blasfeo/include)
endif()

set(SOURCE_FILE solver_function.c)

add_library(solver_function SHARED ${SOURCE_FILE})
target_link_libraries(solver_function PRIVATE ${FATROP_LIB} ${BLASFEO_LIB})
target_compile_options(solver_function PRIVATE -O3 -fPIC)

install(TARGETS solver_function LIBRARY DESTINATION lib)
