#version 430

#define PI 3.1415926535897932384626433832795

layout(local_size_x = 16, local_size_y = 16) in;

uniform float P32, P22;
uniform mat4 invP;
uniform int width, height;
uniform int radius;
uniform float sigma_s, sigma_r;
uniform int vertical;

layout(binding = 0) uniform sampler2D tex_depth;
//layout(binding = 0, r32f) readonly uniform image2D tex_depth;
layout(binding = 1, r32f) writeonly uniform image2D tex_out;

float gaussian(float x){
	return exp(-(x * x) / (2.0 * sigma_s * sigma_s)) / (sqrt(2.0 * PI) * sigma_s);
}

float euclideanLength(float a, float b){
	return exp(-((a - b) * (a - b)) / (2.0 * sigma_r * sigma_r));
}

vec4 backProjection(in vec4 p_ndc){
	float w_c = P32 / (P22 + p_ndc.z);
	vec4 p_clip = w_c * p_ndc;
	return invP * p_clip;
}

vec2 texCoord(in ivec2 imageCoord){
	return (vec2(imageCoord) + vec2(0.5)) / vec2(width, height);
}

void main() {
	ivec2 imageCoord = ivec2(gl_GlobalInvocationID.xy);
	if(imageCoord.x >= width || imageCoord.y >= height){
		return;
	}
	float center = texture(tex_depth, texCoord(imageCoord)).r;
	//float center = imageLoad(tex_depth, imageCoord).r;
	if(center == 1.0){
		imageStore(tex_out, imageCoord, vec4(1.0, 0.0, 0.0, 0.0));
		return;
	}
	vec4 center_ndc = vec4(vec3(texCoord(imageCoord), center) * 2.0 - 1.0, 1.0);
	vec4 center_eye = backProjection(center_ndc);

	float sum = 0.0;
	float factor = 0.0;
	float t = 0.0;
	for (int offset = -radius; offset <= radius; ++offset){
		ivec2 sampleCoord;
		if(vertical == 1){
			sampleCoord = imageCoord + ivec2(0, offset);
		}
		else{
			sampleCoord = imageCoord + ivec2(offset, 0);
		}

		float _sample = texture(tex_depth, texCoord(sampleCoord)).r;
		//float _sample = imageLoad(tex_depth, sampleCoord).r;
		if(_sample == 1.0){
			continue;
		}
		vec4 sample_ndc = vec4(vec3(texCoord(sampleCoord), _sample) * 2.0 - 1.0, 1.0);
		vec4 sample_eye = backProjection(sample_ndc);

		float _offset;
		if(vertical == 1){
			_offset = center_eye.y - sample_eye.y;
		}
		else{
			_offset = center_eye.x - sample_eye.x;
		}
		//factor = gaussian(offset) * euclideanLength(_sample, center);
		factor = gaussian(_offset) * euclideanLength(sample_eye.z, center_eye.z);

		t += factor * _sample;
		sum += factor;
	}

	imageStore(tex_out, imageCoord, vec4(t / sum, 0.0, 0.0, 0.0));
}