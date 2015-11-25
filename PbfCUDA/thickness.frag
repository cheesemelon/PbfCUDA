#version 430

uniform float alpha;
uniform vec2 viewport;

layout(binding = 0) uniform sampler2D tex_depth;

layout(location = 1) out float thickness;

void main(void)
{
	vec2	p = (2.0 * gl_PointCoord - vec2(1.0, 1.0));
	float	r_pow = dot(p, p);

	if(sqrt(r_pow) > 1.0)	discard;

	vec2 texcoord = vec2((gl_FragCoord.x - 0.5) / viewport.x, (gl_FragCoord.y - 0.5) / viewport.y);
	if(alpha == 1.0){
		thickness = 0.05 * (1.0 - texture(tex_depth, texcoord).x);
	}
	else{
		thickness = 0.05;
	}
}
