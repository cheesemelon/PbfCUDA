#version 330

layout (location = 0) in vec2 vertex;

out vec2 texCoord;

void main()
{	
	gl_Position = vec4(vertex, 0.0, 1.0);
	texCoord = (vertex + 1.0) * 0.5;
}