#version 330

uniform float		alpha;
uniform	mat4		P;
uniform	mat4		V;
uniform sampler2D	tex;

layout(location = 0) out vec4	fragColor;
layout(location = 1) out float thickness;
layout(location = 2) out float	smoothed_depth;
layout(location = 3) out float	depth;

void main(void)
{
	vec2	p = (2.0 * gl_PointCoord - vec2(1.0, 1.0));
	float	r_pow = dot(p, p);

	if(sqrt(r_pow) > 1.0)	discard;


//	fragColor = vec4(0.0, 0.0, 0.0, 1.0);
	fragColor = vec4(alpha, alpha, alpha, 1.0);
	//thickness = vec4(alpha, 0.0, 0.0, 1.0);	// alpha = 0.0025 by default


	vec2 uv = vec2(gl_FragCoord.x / 1024.0, gl_FragCoord.y / 1024.0);
	//thickness = 0.05 * (1.0 - r_pow) * (1.0 - texture(tex, uv).x);
	if(alpha == 1.0){
		thickness = 0.05 * (1.0 - texture(tex, uv).x);
	}
	else{
		thickness = 0.05;
	}
}
