#version 330 core

in vec3 UV;

out vec4 FragColor;

uniform samplerCube tex;

void main(){
	FragColor = texture(tex, UV);
}