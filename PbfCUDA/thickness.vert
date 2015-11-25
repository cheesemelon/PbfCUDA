#version 330

uniform mat4 MVP;
uniform float spriteSize;

layout (location = 0) in vec4 vertex;

void
main(void)
{
	//vec4	x_e = V*vec4(vertex.xyz, 1.0);
	//vec4	x_c = P*x_e;
	vec4 x_c = MVP * vec4(vertex.xyz, 1.0);

	float sphereScaleFactor = 1.0 - 0.33*(x_c.z/x_c.w + 1.0); 
	gl_PointSize = 4.0f*spriteSize*sphereScaleFactor;

	gl_Position = x_c;
}
