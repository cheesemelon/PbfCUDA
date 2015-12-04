#version 330

//uniform float		alpha;
uniform	mat4		P;
//uniform	mat4		V;
//uniform vec2		viewport;
//uniform sampler2D	tex;
 
//in vec4	x_c;
in vec4 x_e;
//in mat4 T;
//in mat4 M;

layout(location = 0) out float	smoothed_depth;

void
main(void)
{
	vec2	p = (2.0 * gl_PointCoord - vec2(1.0, 1.0));
	float	r_pow = dot(p, p);

	if(sqrt(r_pow) > 1.0)	discard;

	float z = sqrt(1.0 - r_pow);
	vec3 sphereSurfacePos = vec3(p.x, p.y, z) + x_e.xyz;

	// projection matrix * vertices(eye space coord)
	vec4 clipSpacePos = P * vec4(sphereSurfacePos, 1.0);

	// clip space coord to ndc(normalized device coord), and map the depth range [-1,1] to [0,1]
	float surfaceDepth = 0.5 * (clipSpacePos.z/clipSpacePos.w + 1.0);	

	gl_FragDepth = surfaceDepth;
	smoothed_depth = surfaceDepth;
	//thickness = 0.0;
}


/*
void main(void)
{
	mat4 Vp;
	Vp[0] = vec4(viewport.x/2.0f, 0.0f, 0.0f, 0.0f);
	Vp[1] = vec4(0.0f, viewport.y/2.0f, 0.0f, 0.0f);
	Vp[2] = vec4(0.0f, 0.0f, 0.5f, 0.0f);
	Vp[3] = vec4(viewport.x/2.0f, viewport.y/2.0f, 0.5f, 1.0f);

	mat4 D;
	D[0] = vec4(1, 0, 0, 0);
	D[1] = vec4(0, 1, 0, 0);
	D[2] = vec4(0, 0, 1, 0);
	D[3] = vec4(0, 0, 0, -1);

	mat4 VpPVMT_Inv = x_c.w*inverse(Vp*P*V*M*T);
	//mat4 VpPVMT_Inv = inverse(Vp*P*V*M*T)/gl_FragCoord.w;

	vec4 x_wp = vec4(gl_FragCoord.xy, 0.0f, 1.0f);
	vec4 x_pp = VpPVMT_Inv*x_wp;

	float a = dot(VpPVMT_Inv[2], D*VpPVMT_Inv[2]);
	float b = dot(x_pp, D*VpPVMT_Inv[2]);
	float c = dot(x_pp, D*x_pp);

	if(b*b - a*c < 0)
		discard;

	//if(gl_FragCoord.y > viewport.y/2 || gl_FragCoord.x > viewport.x/2)
	//	discard;

	depth = 0.0;
	gl_FragDepth = 0.0;	

	//else
	//	discard;
}
*/
