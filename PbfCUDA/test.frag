#version 430

uniform	mat4 projectionMatrix;
uniform	mat4 invProjectionMatrix;
uniform vec2 viewport;
//uniform float near;
uniform int renderType;
uniform float	alpha;

in vec2 texCoord;

layout(binding = 0) uniform sampler2D tex_depth;
layout(binding = 1) uniform sampler2D tex_thickness;
layout(binding = 2) uniform sampler1D tex_diffuseWrap;
layout(binding = 3) uniform samplerCube tex_cubemap;

layout(location = 0) out vec4 fragColor;

vec4 getEyePos(vec4 Xndc)
{
	float w_c = projectionMatrix[3][2]/(projectionMatrix[2][2] + Xndc.z);
	vec4 Xc = w_c*Xndc;
	vec4 Xe = invProjectionMatrix*Xc;

	return Xe;
}

vec3 calcNormalVector(vec2 coord, vec2 texelSize, out float gradient)
{
	float z_ndc = texture(tex_depth, coord).x;
	z_ndc = 2.0*z_ndc - 1.0;
	vec4 Xndc = vec4(2.0*coord - 1.0, z_ndc, 1.0);
	vec4 Xe = getEyePos(Xndc);

	z_ndc = texture(tex_depth, coord + vec2(texelSize.x, 0.0)).x;
	z_ndc = 2.0*z_ndc - 1.0;
	Xndc = vec4(2.0*(coord + vec2(texelSize.x, 0.0)) - 1.0, z_ndc, 1.0);
	vec3 dx = (getEyePos(Xndc) - Xe).xyz;
	
	z_ndc = texture(tex_depth, coord + vec2(-texelSize.x, 0.0)).x;
	z_ndc = 2.0*z_ndc - 1.0;
	Xndc = vec4(2.0*(coord + vec2(-texelSize.x, 0.0)) - 1.0, z_ndc, 1.0);
	vec3 dx2 = (Xe - getEyePos(Xndc)).xyz;

	z_ndc = texture(tex_depth, coord + vec2(0.0, texelSize.y)).x;
	z_ndc = 2.0*z_ndc - 1.0;
	Xndc = vec4(2.0*(coord + vec2(0.0, texelSize.y)) - 1.0, z_ndc, 1.0);
	vec3 dy = (getEyePos(Xndc) - Xe).xyz;
	
	z_ndc = texture(tex_depth, coord + vec2(0.0, -texelSize.y)).x;
	z_ndc = 2.0*z_ndc - 1.0;
	Xndc = vec4(2.0*(coord + vec2(0.0, -texelSize.y)) - 1.0, z_ndc, 1.0);
	vec3 dy2 = (Xe - getEyePos(Xndc)).xyz;

	// gradient for outline
	gradient = abs(dx.z - dx2.z) + abs(dy.z - dy2.z);
	gradient = 1.0 - clamp(gradient, 0.0, 1.0);

	if( abs(dx.z) > abs(dx2.z) )		dx = dx2;
	if( abs(dy.z) > abs(dy2.z) )		dy = dy2;

	return normalize(cross(dx, dy));
}

void main()
{
	vec2 texelSize = vec2(1.0/viewport.x, 1.0/viewport.y);
	float depth = texture(tex_depth, texCoord).x;
	if(depth > 0.9999) { discard; }

	float thickness = min(texture(tex_thickness, texCoord).x, 1.0);
	float gradient;

	// shading
	vec3 n, v, l, r;
	n = calcNormalVector(texCoord, texelSize, gradient);
	v = normalize(vec3(0.0, 0.0, 1.0));
	l = normalize(vec3(1.0, 1.0, 1.0));
	//l = normalize(vec3(0.0, 0.0, 1.0));
	r = normalize(2.0*dot(n, l)*n - l);

	// depth image
	if(renderType == 0)
	{
		fragColor = vec4(vec3(depth), 1.0);
	}
	// thickness image
	else if(renderType == 1)
	{
		fragColor = vec4(vec3(thickness), 1.0);
	}
	// simple shading
	else if(renderType == 2)
	{
		vec3 ambient = vec3(0.1);
		vec3 diffuse = 1.0*(0.5*max(dot(n, l), 0.0) + 0.5) * vec3(1.0);
		vec3 specular = 1.0*pow(max(dot(v, r), 0.0), 50.0) * vec3(1.0);

		fragColor = vec4(diffuse + specular, 1.0);
	}
	// test shading
	else if(renderType == 3)
	{
		//vec3 ambient = 0.1*vec3(1.0, 1.0, 1.0);
		vec3 diffuse = 1.0*(0.5*max(dot(n, l), 0.0) + 0.5)*vec3(0.5 - thickness, 1.0 - thickness, 1.6 - thickness);
        vec3 specular = 1.0*pow(max(dot(v, r), 0.0), 10.0)*vec3(1.0, 1.0, 1.0);

		fragColor = vec4(diffuse+ specular, 1.0);
	}
	// test shading + reflection + refraction
	else if(renderType == 4)
	{
		vec3 diffuse = vec3(0.5 * max(dot(n, l), 0.0) + 0.5);
        vec3 specular = 1.0*pow(max(dot(v, r), 0.0), 10.0)*vec3(1.0, 1.0, 1.0);
		vec3 p = vec3((texCoord - 0.5) * 0.5, depth);
		vec3 eye = normalize(p - v);
		
		vec3 reflection = texture(tex_cubemap, reflect(eye, n)).rgb;
		reflection = 0.2 * reflection;
		vec3 refraction = texture(tex_cubemap, refract(eye, n, 0.685)).rgb;
		//refraction = refraction * (1.0 - thickness);
		fragColor = vec4((diffuse * refraction) + reflection + specular, thickness + length(specular));
	}
	// cel shading
	else
	{
		float ambientCoeff = 0.1;
		float diffuseCoeff = 1.0 * (0.5 * max(dot(n, l), 0.0)); // half lambertian
		float specularCoeff = 1.0 * pow(max(dot(v, r), 0.0), 50.0);
		float intensity = clamp(ambientCoeff + diffuseCoeff + specularCoeff, 0.0, 1.0);

		// map intensity to 1D texture
		vec3 mappedColor = texture(tex_diffuseWrap, intensity).rgb;

		// Add outline by multiplying gradient
		fragColor = vec4(gradient * mappedColor, 1.0);
	}
}
