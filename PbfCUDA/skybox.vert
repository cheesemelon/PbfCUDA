#version 330 core

layout (location = 0) in vec2 vertex;

out vec3 texCoord;

uniform mat3 invV;
uniform mat4 invP;

void main(){
	vec3 p_ndc = vec3(vertex, 0.0);
	texCoord = invV * (invP * vec4(p_ndc, 1.0)).xyz;
	gl_Position = vec4(p_ndc, 1.0);
}