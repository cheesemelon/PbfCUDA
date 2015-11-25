#version 430

uniform	mat4 projectionMatrix;
uniform	mat4 invProjectionMatrix;
uniform vec2 viewport;
//uniform float near;
uniform int renderType;
uniform float	alpha;
uniform int genNormal;

in vec2 texCoord;

layout(binding = 0) uniform sampler2D tex_depth;
layout(binding = 1) uniform sampler2D tex02;
layout(binding = 2) uniform sampler1D diffuseWrap;
layout(binding = 3) uniform sampler2D outlineTexture;
layout(binding = 4) uniform samplerCube cubemapTexture;
layout(binding = 5) uniform sampler2D tex_normal_x;
layout(binding = 6) uniform sampler2D tex_normal_y;
layout(binding = 7) uniform sampler2D tex_normal_z;

layout(location = 0) out vec4 fragColor;

vec4 getEyePos(vec4 Xndc)
{
	float w_c = projectionMatrix[3][2]/(projectionMatrix[2][2] + Xndc.z);
	vec4 Xc = w_c*Xndc;
	vec4 Xe = invProjectionMatrix*Xc;

	return Xe;
}

vec3 calcNormalVector(vec2 coord, vec2 texelSize)
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

	if( abs(dx.z) > abs(dx2.z) )
		dx = dx2;
	//else if( dx.z == dx2.z )
	//	dx = vec3(1.0, 0.0, 0.0);

	if( abs(dy.z) > abs(dy2.z) )
		dy = dy2;
	//else if( abs(dy.z) > abs(dy2.z) )
	//	//dy = (dy + dy2) * 0.5;
	//	dy = vec3(0.0, 1.0, 0.0);

	return normalize(cross(dx, dy));
}

vec3 calcNormalVector2(vec2 coord, vec2 texelSize){
	float diff = 0.5;

	float z_ndc = texture(tex_depth, coord).x;
	vec3 center = 2.0 * vec3(coord, z_ndc) - 1.0;
	center = getEyePos(vec4(center, 1.0)).xyz;

	z_ndc = texture(tex_depth, coord + vec2(-texelSize.x, 0.0)).x;
	vec3 left = 2.0 * vec3(coord + vec2(-texelSize.x, 0.0), z_ndc) - 1.0;
	left = getEyePos(vec4(left, 1.0)).xyz;
	
	z_ndc = texture(tex_depth, coord + vec2(texelSize.x, 0.0)).x;
	vec3 right = 2.0 * vec3(coord + vec2(texelSize.x, 0.0), z_ndc) - 1.0;
	right = getEyePos(vec4(right, 1.0)).xyz;

	vec3 right_direction = right - left;
	//if(abs(abs(center.z) - abs(left.z)) > diff){
	//	right_direction = right - center;
	//}
	//else if(abs(abs(center.z) - abs(right.z)) > diff){
	//	right_direction = center - left;
	//}

	z_ndc = texture(tex_depth, coord + vec2(0.0, texelSize.y)).x;
	vec3 up = 2.0 * vec3(coord + vec2(0.0, texelSize.y), z_ndc) - 1.0;
	up = getEyePos(vec4(up, 1.0)).xyz;

	z_ndc = texture(tex_depth, coord + vec2(0.0, -texelSize.y)).x;
	vec3 down = 2.0 * vec3(coord + vec2(0.0, -texelSize.y), z_ndc) - 1.0;
	down = getEyePos(vec4(down, 1.0)).xyz;

	vec3 up_direction = up - down;
	//if(abs(abs(center.z) - abs(up.z)) > diff){
	//	up_direction = center - down;
	//}
	//if(abs(abs(center.z) - abs(down.z)) > diff){
	//	up_direction = up - center;
	//}

	vec3 res = normalize(cross(right_direction, up_direction));
	if(res.z < 0){
		return normalize(vec3(res.xy, 0));
	}
	return res;
}

void main()
{
	vec2 texelSize = vec2(1.0/viewport.x, 1.0/viewport.y);
	float depth = texture(tex_depth, texCoord).x;
	if(depth > 0.9999) discard;

	float thickness = clamp(texture(tex02, texCoord).x, 0.0, 1.0);
	//thickness = 0.8 * thickness;	// Thickness attenuation. Why?
	//thickness = 0.0;

	// shading
	vec3 n, v, l, r;
	//n = vec3(texture(tex_normal_x, texCoord).x,
	//	texture(tex_normal_y, texCoord).x,
	//	texture(tex_normal_z, texCoord).x);
	//n = normalize(n);
	if(genNormal == 0){
		n = calcNormalVector(texCoord, texelSize);
	}
	else{
		n = calcNormalVector2(texCoord, texelSize);
	}
	v = normalize(vec3(0.0, 0.0, 1.0));
	l = normalize(vec3(1.0, 1.0, 1.0));
	//l = normalize(vec3(0.0, 0.0, 1.0));
	r = normalize(2.0*dot(n, l)*n - l);

	if(renderType == 0)
	{
		// depth image
		//fragColor = vec4(depth * vec3(0.5, 0.5, 0.5), 1.0);
		fragColor = vec4(vec3(depth), 1.0);
		//fragColor = vec4(vec3(dot(n, vec3(0, 0, 1))), 1.0);

		//vec3 center = 2.0 * vec3(texCoord, depth) - 1.0;
		//center = getEyePos(vec4(center, 1.0)).xyz;
		//fragColor = vec4(vec3(-center.z / 60.0), 1.0);
	}
	else if(renderType == 1) 
	{
		// thickness image
		fragColor = vec4(vec3(thickness), 1.0);
	}
	else if(renderType == 2) 
	{
		// simple shading
		//vec3 ambient = 0.1*vec3(1.0, 1.0, 1.0);
		//vec3 diffuse = 1.0*(0.5*max(dot(n, l), 0.0) + 0.5)*vec3(1.0, 1.0, 1.0);
		//vec3 specular = 1.0*pow(max(dot(v, r), 0.0), 50.0)*vec3(1.0, 1.0, 1.0);

		//fragColor = vec4(diffuse + specular, 1.0);
		fragColor = vec4(vec3(max(dot(n, l), 0.0)), 1.0);
		//fragColor = vec4(vec3(n), 1.0);
	}
	else if(renderType == 3)
	{
		// final rendering
		//vec3 ambient = 0.1*vec3(1.0, 1.0, 1.0);
		vec3 diffuse = 1.0*(0.5*max(dot(n, l), 0.0) + 0.5)*vec3(0.5 - thickness, 1.0 - thickness, 1.6 - thickness);
        vec3 specular = 1.0*pow(max(dot(v, r), 0.0), 10.0)*vec3(1.0, 1.0, 1.0);

		fragColor = vec4(diffuse+ specular, 1.0);
	}
	else if(renderType == 4){
		// final rendering + reflection + refraction
		//vec3 ambient = 0.1*vec3(1.0, 1.0, 1.0);
		vec3 diffuse = vec3(0.5 * max(dot(n, l), 0.0) + 0.5);
		//vec3 diffuse = 1.0*(0.5*max(dot(n, l), 0.0) + 0.5)*vec3(0.5 - thickness, 1.0 - thickness, 1.6 - thickness);
        vec3 specular = 1.0*pow(max(dot(v, r), 0.0), 10.0)*vec3(1.0, 1.0, 1.0);
		vec3 p = vec3((texCoord - 0.5) * 0.5, depth);
		vec3 eye = normalize(p - v);
		//float transparency = thickness + length(specular) + 0.5;
		//transparency = clamp(transparency, 0.0 ,1.0);
		//float transparency = thickness * 0.5 + 0.5;
		float outline = texture(outlineTexture, texCoord).r;
		
		vec3 reflection = texture(cubemapTexture, reflect(eye, n)).rgb;
		reflection = 0.2 * reflection;
		vec3 refraction = texture(cubemapTexture, refract(eye, n, 0.685)).rgb;
		//refraction = refraction * (1.0 - thickness);
		fragColor = vec4((diffuse * refraction) + reflection + specular, thickness + length(specular));
	}
	else if(renderType == 5) 
	{
		// outline
		fragColor = vec4(1, 0, 0, 1) * vec4(texture(outlineTexture, texCoord).r);
	}
	else{
		// cel shading

		float ambientCoeff = 0.1;
		//float diffuseCoeff = 1.0 * (0.5 * max(dot(n, l), 0.0)) - thickness;
		float diffuseCoeff = 1.0 * (0.5 * max(dot(n, l), 0.0));
		float specularCoeff = 1.0 * pow(max(dot(v, r), 0.0), 50.0);
		float intensity = clamp(ambientCoeff + diffuseCoeff + specularCoeff, 0.0, 1.0);

		vec3 result = texture(diffuseWrap, intensity).rgb;
		//vec3 result = texture(diffuseWrap, intensity * (1.0 - thickness)).rgb;// * (1.0 - thickness);

		//fragColor = vec4(texture(diffuseWrap, intensity).rgb, 1.0);
		
		float outline = texture(outlineTexture, texCoord).r;
		vec4 outlineColor = vec4(1.0, 0.0, 0.0, outline);
		fragColor = vec4(result, alpha);
		fragColor = outlineColor * outlineColor.a + fragColor * (1 - outlineColor.a);
	}
}
