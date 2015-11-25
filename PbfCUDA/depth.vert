#version 330

uniform mat4 V;
uniform mat4 P;
//uniform vec2 viewport;
uniform float spriteSize;

layout (location = 0) in vec4 vertex;
//layout (location = 1) in mat4 mat;

//out vec4 x_c;
out vec4 x_e;
//out mat4 T;
//out mat4 M;

void
main(void)
{
			x_e = V * vec4(vertex.xyz, 1.0);
	vec4	x_c = P * x_e;

	float sphereScaleFactor = 1.0 - 0.33 * (x_c.z/x_c.w + 1.0); 
	gl_PointSize = 4.0f * spriteSize * sphereScaleFactor;

	gl_Position = x_c;
}

/*
void main(void)
{
	// op.[] parameter: comlumn index
	T[0] = vec4(1, 0, 0, 0);
	T[1] = vec4(0, 1, 0, 0);
	T[2] = vec4(0, 0, 1, 0);
	T[3] = vec4(0, 0, 0, 1);

	M[0] = vec4(12, 0, 0, 0);
	M[1] = vec4(0, 6, 0, 0);
	M[2] = vec4(0, 0, 6, 0);
	M[3] = vec4(vertex.xyz, 1);
	
	mat4 D;
	D[0] = vec4(1, 0, 0, 0);
	D[1] = vec4(0, 1, 0, 0);
	D[2] = vec4(0, 0, 1, 0);
	D[3] = vec4(0, 0, 0, -1);

	mat4 PVMT = P*V*M*T;
	mat4 PVMT_T = transpose(PVMT);
	
	x_c = PVMT*vec4(0, 0, 0, 1);

	float a = dot(PVMT_T[3], D*PVMT_T[3]);
	float b = dot(PVMT_T[0], D*PVMT_T[3]);
	float c = dot(PVMT_T[0], D*PVMT_T[0]);

	float x1 = (b - sqrt(b*b - a*c))/a;
	float x2 = (b + sqrt(b*b - a*c))/a;

	a = dot(PVMT_T[3], D*PVMT_T[3]);
	b = dot(PVMT_T[1], D*PVMT_T[3]);
	c = dot(PVMT_T[1], D*PVMT_T[1]);

	float y1 = (b - sqrt(b*b - a*c))/a;
	float y2 = (b + sqrt(b*b - a*c))/a;
	
	float BBSize = 0;
	float x_length = abs(x2 - x1);
	float y_length = abs(y2 - y1);

	if(x_length > y_length)
		BBSize = (viewport.x/2.0f)*(x_length/x_c.w);
	else
		BBSize = (viewport.y/2.0f)*(y_length/x_c.w);

	gl_PointSize = BBSize;

	gl_Position = x_c;
	//gl_Position = vec4(dot(PVMT_T[0], D*PVMT_T[3]), dot(PVMT_T[1], D*PVMT_T[3]), 0, dot(PVMT_T[3], D*PVMT_T[3]));
}
*/
