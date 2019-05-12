#ifndef SMANDELBROTR_APP_VERTS_H
#define SMANDELBROTR_APP_VERTS_H

#include "appVbo.h"

#include <SFML/OpenGL.hpp>

class AppVerts {
    GLuint mId;
    AppVbo *mPositionVbo, *mTexCoordsVbo;
    int mNumVerts;
public:
AppVerts(int numAttrs, int numVerts, GLuint posAttr, float *posCoords, int texAttr, float *texCoords) : mNumVerts(numVerts) {
        // NOTE all VBOs are 2-component (X,Y or S,T)
        glGenVertexArrays(1, &mId);
        glBindVertexArray(mId);
        mPositionVbo = new AppVbo(numAttrs, posAttr, posCoords);
        mTexCoordsVbo = new AppVbo(numAttrs, texAttr, texCoords);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    void draw() {
        glDrawArrays(GL_TRIANGLE_STRIP, 0, mNumVerts);
    }
    void updatePosition(float *newCoords) {
        mPositionVbo->update(newCoords);
    }
    void updateTexCoords(float *newCoords) {
        mTexCoordsVbo->update(newCoords);
    }
    // bind this VertexArray so OpenGL can use it
    void bind() {
        glBindVertexArray(mId);
    }
    // unbind this VertexArray so OpenGL will stop using it
    void unbind() {
        glBindVertexArray(0);
    }
};

#endif
