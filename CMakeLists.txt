cmake_minimum_required(VERSION 3.10)

# On Windows, these variables need to be customized for your environment.
# On Linux you can apt-get install these to system locations
if(WIN32)
    set(SDL2_DIR          "/Users/rallen/Documents/Devel/SDL2-2.0.9")
    set(GLEW_INCLUDE_DIR  "/Users/rallen/Documents/Devel/glew-2.1.0/include")
    set(GLEW_LIBRARY      "/Users/rallen/Documents/Devel/glew-2.1.0/lib/Release/x64/glew32.lib")
    set(GLEW_DLL_DIR      "/Users/rallen/Documents/Devel/glew-2.1.0/bin/Release/x64")
    set(GLM_DIR           "/Users/rallen/Documents/Devel/glm-0.9.9.5")
endif()

include(CheckLanguage)
check_language(CUDA)

project(smandelbrotr
    VERSION     0.1
    DESCRIPTION "SDL2 CUDA OpenGL Mandelbrot Viewer"
    LANGUAGES   CUDA CXX
)

add_executable(smandelbrotr app.cpp  appCUDA.cpp  appGL.cpp  main.cpp  
                            mandelbrot.cu mandelbrotKernels.cu appCUDAProgram.cu)

# CUDA wants C++11
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
# (don't know why I needed to add this explicitly)
target_link_libraries(smandelbrotr cuda nvrtc)

set_target_properties(smandelbrotr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(smandelbrotr PUBLIC cxx_std_11)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(smandelbrotr PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(smandelbrotr PUBLIC "${GLM_DIR}")

# Windows prereq: get dev libraries from https://www.libsdl.org/download-2.0.php
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
# workaround for CMP0004 issue with the SDL2 libraries on Linux
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)
target_link_libraries(smandelbrotr ${SDL2_LIBRARIES})

# OpenGL
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
if(NOT TARGET OpenGL::OpenGL)
  target_link_libraries(smandelbrotr OpenGL::GL)
  target_include_directories(smandelbrotr PUBLIC OpenGL::GL)
else()
  target_link_libraries(smandelbrotr OpenGL::OpenGL)
  target_include_directories(smandelbrotr PUBLIC OpenGL::OpenGL)
endif()

# GL Extension Wrangler (GLEW) get dev libraries from http://glew.sourceforge.net/
find_package(GLEW REQUIRED)
target_link_libraries(smandelbrotr "${GLEW_LIBRARIES}")
target_include_directories(smandelbrotr PUBLIC "${GLEW_INCLUDE_DIRS}")

# Windows: copy the dll as a post-build step
if(WIN32)
    add_custom_command(TARGET smandelbrotr POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SDL2_DIR}/lib/x64/SDL2.dll"
            $<TARGET_FILE_DIR:smandelbrotr>)
    add_custom_command(TARGET smandelbrotr POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLEW_DLL_DIR}/glew32.dll"
            $<TARGET_FILE_DIR:smandelbrotr>)
endif()
