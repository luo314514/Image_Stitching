cmake_minimum_required(VERSION 3.10)
project(OBJ_GSP)

set(CMAKE_CXX_STANDARD 14)

# 1. 寻找零件库
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED) # 【新增】寻找 OpenMP

# 2. 包含头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include/eigen3)
include_directories(.)
include_directories(Debugger)
include_directories(Feature)
include_directories(Mesh)
include_directories(Stitching)
include_directories(Util)

# 3. 搜集源码
file(GLOB SRC_FILES_ROOT "*.cpp")
file(GLOB SRC_FILES_DEBUGGER "Debugger/*.cpp")
file(GLOB SRC_FILES_FEATURE "Feature/*.cpp")
file(GLOB SRC_FILES_MESH "Mesh/*.cpp")
file(GLOB SRC_FILES_STITCHING "Stitching/*.cpp")
file(GLOB SRC_FILES_UTIL "Util/*.cpp")

set(SRC_FILES 
    ${SRC_FILES_ROOT} 
    ${SRC_FILES_DEBUGGER} 
    ${SRC_FILES_FEATURE} 
    ${SRC_FILES_MESH} 
    ${SRC_FILES_STITCHING} 
    ${SRC_FILES_UTIL}
)

# 4. 编译可执行文件
add_executable(obj_gsp ${SRC_FILES})

# 5. 【关键】链接所有需要的库
# 加上 OpenMP::OpenMP_CXX 即可解决最后的报错
target_link_libraries(obj_gsp ${OpenCV_LIBS} vl gflags OpenMP::OpenMP_CXX)