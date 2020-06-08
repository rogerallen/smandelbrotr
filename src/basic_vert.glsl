#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoords;
out vec4 vTexCoords;

uniform mat4 cameraToView;

void main()
{
    gl_Position = cameraToView * vec4(position, 1.0);
    vTexCoords = vec4(texCoords, 0.0, 1.0);
}
