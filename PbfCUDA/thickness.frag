#version 430

uniform float alpha;

layout(location = 0) out float thickness;

void main(void)
{
	vec2	p = (2.0 * gl_PointCoord - vec2(1.0, 1.0));
	float	r_pow = dot(p, p);

	if(sqrt(r_pow) > 1.0)	discard;

	thickness = alpha;
}
