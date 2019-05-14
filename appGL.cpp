//#include "appGL.h"
#include <SFML/OpenGL.hpp>
#include <stdio.h>

void GLAPIENTRY
MessageCallback( GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam )
{
    if(severity != GL_DEBUG_SEVERITY_NOTIFICATION) {
        fprintf( stderr,
                 "GL CALLBACK: %s %stype = 0x%x, severity = 0x%x, message = %s\n",
                 (( severity == GL_DEBUG_SEVERITY_HIGH ) ? "HIGH" :
                  ( severity == GL_DEBUG_SEVERITY_MEDIUM ) ? "MEDIUM" :
                  ( severity == GL_DEBUG_SEVERITY_LOW ) ? "LOW" : "NOTIFICATION"),
                 ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR ** " : "" ),
             type, severity, message );
    }
}
