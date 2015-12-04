#version 330 core

layout (location = 0) in vec2 vertex;

out vec3 texCoord;

uniform mat4 V;
uniform mat4 P;

void main(){
	//mat4 inverseProjection = transpose(inverse(P));
	//mat3 inverseModelView = transpose(inverse(mat3(V)));
	//vec3 unprojected = (vec4(vertices, 1.0) * inverseProjection).xyz;
	//texCoord = unprojected * inverseModelView;
	//gl_Position = vec4(vertices, 1.0);

	vec3 p_ndc = vec3(vertex, 0.0);
	mat4 inverseProjection = inverse(P);
	mat3 inverseModelView = inverse(mat3(V));
	texCoord = inverseModelView * (inverseProjection * vec4(p_ndc, 1.0)).xyz;
	gl_Position = vec4(p_ndc, 1.0);
}