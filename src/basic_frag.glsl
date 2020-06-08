#version 330
uniform sampler2D texture;
in vec4 vTexCoords;
out vec4 fColor;

void main()
{
    fColor = texture2D(texture, vTexCoords.st);
}
