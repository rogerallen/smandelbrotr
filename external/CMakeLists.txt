# Download and unzip all external libraries
#
# found at https://www.scivision.dev/cmake-download-verify-file/
function(download_file url filename hash_type hash)

if(NOT EXISTS ${filename})
  message("downloading " ${url})
  file(DOWNLOAD ${url} ${filename}
       TIMEOUT 60  # seconds
       EXPECTED_HASH ${hash_type}=${hash}
       TLS_VERIFY ON)
endif()

endfunction(download_file)

function(extract_zip zipfile outfile)

if(NOT EXISTS ${outfile})
  message("unzipping   " ${outfile})
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${zipfile} 
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

endfunction(extract_zip)

# ======================================================================
# SDL (Windows Only)
if(WIN32)
    set( SDL_URL    https://www.libsdl.org/release/SDL2-devel-2.0.12-VC.zip )
    set( SDL_ZIP    ${CMAKE_CURRENT_SOURCE_DIR}/SDL2-devel-2.0.12-VC.zip )
    set( SDL_ROOT   ${CMAKE_CURRENT_SOURCE_DIR}/SDL2-2.0.12 )
    set( SDL_MD5    c2262a2f2d4e8487b75b541bfee3c2c4 )
    download_file( ${SDL_URL} ${SDL_ZIP} MD5 ${SDL_MD5} )
    extract_zip( ${SDL_ZIP} ${SDL_ROOT} )
    if(NOT EXISTS ${SDL_ROOT}/sdl2-config.cmake)
       file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/sdl2-config.cmake DESTINATION ${SDL_ROOT}) 
    endif()
endif()

# ======================================================================
# GLEW (Windows Only)
if(WIN32)
    set( GLEW_URL   https://github.com/nigels-com/glew/releases/download/glew-2.1.0/glew-2.1.0-win32.zip )
    set( GLEW_ZIP   ${CMAKE_CURRENT_SOURCE_DIR}/glew-2.1.0.zip )
    set( GLEW_ROOT  ${CMAKE_CURRENT_SOURCE_DIR}/glew-2.1.0 )
    set( GLEW_MD5   32a72e6b43367db8dbea6010cd095355 )
    download_file( ${GLEW_URL} ${GLEW_ZIP} MD5 ${GLEW_MD5} )
    extract_zip( ${GLEW_ZIP} ${GLEW_ROOT} )
endif()

# ======================================================================
# GLM
set( GLM_URL    https://github.com/g-truc/glm/archive/0.9.9.8.zip )
set( GLM_ZIP    ${CMAKE_CURRENT_SOURCE_DIR}/glm-0.9.9.8.zip )
set( GLM_ROOT   ${CMAKE_CURRENT_SOURCE_DIR}/glm-0.9.9.8 )
set( GLM_MD5    a421a37d1896ab74765c3f94431ca0f6 )
download_file( ${GLM_URL} ${GLM_ZIP} MD5 ${GLM_MD5} )
extract_zip( ${GLM_ZIP} ${GLM_ROOT} )

# ======================================================================
# IMGUI
set( IMGUI_URL  https://github.com/ocornut/imgui/archive/v1.76.zip )
set( IMGUI_ZIP  ${CMAKE_CURRENT_SOURCE_DIR}/imgui-1.76.zip )
set( IMGUI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/imgui-1.76 )
set( IMGUI_MD5  f86369c278a9f34b07b8ed09c3a93b6a )
download_file( ${IMGUI_URL} ${IMGUI_ZIP} MD5 ${IMGUI_MD5} )
extract_zip( ${IMGUI_ZIP} ${IMGUI_ROOT} )

