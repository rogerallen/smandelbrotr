#ifndef SMANDELBROTR_APP_VBO_H
#define SMANDELBROTR_APP_VBO_H

#include <GL/glew.h>

#define BUFFER_OFFSET(i) ((void*)(i))

class AppVbo {
    GLuint mId;
    int mNumAttrs;
public:
    AppVbo(int numAttrs, int attr, float *data) : mNumAttrs(numAttrs) {
        glGenBuffers(1, &mId);
        glBindBuffer(GL_ARRAY_BUFFER, mId);
        glBufferData(GL_ARRAY_BUFFER, numAttrs*sizeof(float), data, GL_DYNAMIC_DRAW);
        // NOTE: fixed to 2 float components
        glEnableVertexAttribArray(attr);
        glVertexAttribPointer(attr, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
    }
    void update(float *data) {
        // data better be the same length as mNumAttrs!
        glBindBuffer(GL_ARRAY_BUFFER, mId);
        glBufferSubData(GL_ARRAY_BUFFER, 0, mNumAttrs*sizeof(float), data);
    }
};

#endif
