SMandelbrotr
============

SDL2 CUDA OpenGL Mandelbrot explorer.

*WORK IN PROGRESS*


Building
--------

Windows
-------

*FIXME -- add GLM*

Most of this is derived from https://trenki2.github.io/blog/2017/06/02/using-sdl2-with-cmake/  (Thanks!)

* download SDL2 dev libraries from https://www.libsdl.org/download-2.0.php, unzip and put folder within somewhere accessible.
* set var SDL2_DIR in CMakeLists.txt (or your environment) to that directory
* add sdl2-config.cmake to SDL2_DIR containing
```
set(SDL2_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")

# Support both 32 and 64 bit builds
if (${CMAKE_SIZEOF_VOID_P} MATCHES 8)
  set(SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/lib/x64/SDL2.lib;${CMAKE_CURRENT_LIST_DIR}/lib/x64/SDL2main.lib")
else ()
  set(SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/lib/x86/SDL2.lib;${CMAKE_CURRENT_LIST_DIR}/lib/x86/SDL2main.lib")
endif ()

string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)
```

* download GLEW dev libraries from http://glew.sourceforge.net/, unzip and put folder within somewhere accessible
* see CmakeLists.txt for how to set GLEW_INCLUDE_DIR, GLEW_LIBRARY and GLEW_DLL_DIR

* then CMake build & Project build and you should be good.


Linux
-----

`sudo apt-get install libsdl2-dev libglew-dev libglm-dev`

Build
-----

```
mkdir Debug
cd Debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

```
mkdir Release
cd Release
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
