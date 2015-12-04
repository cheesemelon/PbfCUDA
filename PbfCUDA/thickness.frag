#version 430

uniform float alpha;
uniform vec2 viewport;
uniform float P32, P22;
uniform float intensityRange;

layout(binding = 0) uniform sampler2D tex_depth;

layout(location = 0) out float thickness;

float backProjection(in float z_ndc){
	return -P32 / (P22 + z_ndc);
}

void main(void)
{
	vec2	p = (2.0 * gl_PointCoord - vec2(1.0, 1.0));
	float	r_pow = dot(p, p);

	if(sqrt(r_pow) > 1.0)	discard;

	float depth = texture(tex_depth, vec2(gl_FragCoord.x / viewport.x, gl_FragCoord.y / viewport.y)).x;
	depth = 1.0 - (backProjection(depth) / intensityRange);
	//if(alpha == 1.0){
		//thickness = 0.05 * depth;
		thickness = 0.05;
	//}
	//else{
	//	thickness = 0.05;
	//}
}
