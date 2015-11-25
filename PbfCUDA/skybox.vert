#version 330 core

layout (location = 0) in vec3 vertices;

out vec3 UV;

uniform mat4 V;
uniform mat4 P;

void main(){
	mat4 inverseProjection = transpose(inverse(P));
	mat3 inverseModelView = transpose(inverse(mat3(V)));
	vec3 unprojected = (vec4(vertices, 1.0) * inverseProjection).xyz;
	UV = unprojected * inverseModelView;
	gl_Position = vec4(vertices, 1.0);
}