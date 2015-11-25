#version 330

uniform	mat4 P;
uniform	mat4 V;

layout (location = 0) in vec3 vertex;
layout (location = 1) in vec2 inTexCoord;

out vec2 texCoord;

void main()
{	
	texCoord = inTexCoord;
	gl_Position =  P*V*vec4(vertex, 1.0);
}