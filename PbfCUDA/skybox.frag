#version 430 core

in vec3 texCoord;

out vec4 FragColor;

layout(binding = 0) uniform samplerCube tex_cubemap;

void main(){
	FragColor = texture(tex_cubemap, texCoord);
}