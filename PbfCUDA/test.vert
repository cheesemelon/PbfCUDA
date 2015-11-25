#version 330

uniform	mat4 curProjectionMatrix;
uniform	mat4 curModelViewMatrix;

layout (location = 0) in vec3 vertex;
layout (location = 1) in vec2 inTexCoord;

out vec2 texCoord;

void main()
{	
	texCoord = inTexCoord;
	vec4 pos = curProjectionMatrix*curModelViewMatrix*vec4(vertex, 1.0);
	gl_Position = pos;
}