#version 430

#define PI 3.1415926535897932384626433832795

layout(local_size_x = 16, local_size_y = 16) in;

uniform float P32, P22;			// Projection matrix componet [3][2], [2][2].
uniform mat4 invP;				// Inverse projection matrix.
uniform int width, height;
uniform int radius;				// Kernel radius.
uniform float sigma_s, sigma_r;
uniform int vertical;			// boolean to check that filter runs vertically.

uniform int depthCorrection;	// boolean for depth correction.
uniform int filter_3D;			// boolean for 3D filtering.

layout(binding = 0) uniform sampler2D tex_depth;
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

//@ Get image space coordinates from window coordinates. range in [0.0, 1.0]
vec2 texCoord(in ivec2 imageCoord){
	return (vec2(imageCoord) + vec2(0.5)) / vec2(width, height);
}

void main() {
	// Calc thread ID(index).
	ivec2 imageCoord = ivec2(gl_GlobalInvocationID.xy);
	if(imageCoord.x >= width || imageCoord.y >= height){
		return;
	}

	// Get center(current) value.
	float center = texture(tex_depth, texCoord(imageCoord)).r;
	if(center == 1.0){
		imageStore(tex_out, imageCoord, vec4(1.0, 0.0, 0.0, 0.0));
		return;
	}
	vec4 center_ndc = vec4(vec3(texCoord(imageCoord), center) * 2.0 - 1.0, 1.0);
	vec4 center_eye = backProjection(center_ndc);

	// Run filter.
	float sum = 0.0;
	float factor = 0.0;
	float t = 0.0;
	for (int offset = -radius; offset <= radius; ++offset){
		// Calc sample index.
		ivec2 sampleCoord;
		if(vertical == 1){
			sampleCoord = imageCoord + ivec2(0, offset);
		}
		else{
			sampleCoord = imageCoord + ivec2(offset, 0);
		}

		// Get sample value.
		float _sample = texture(tex_depth, texCoord(sampleCoord)).r;
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

		// Calc spatial weight
		float s_factor;
		if (filter_3D == 1){
			s_factor = gaussian(_offset);
		}
		else{
			s_factor = gaussian(offset);
		}

		// Calc range weight
		float r_factor;
		if (depthCorrection == 1){
			r_factor = euclideanLength(sample_eye.z, center_eye.z);
		}
		else{
			r_factor = euclideanLength(_sample, center);
		}
		factor = s_factor * r_factor;

		//factor = gaussian(_offset) * euclideanLength(sample_eye.z, center_eye.z);
		t += factor * _sample;
		sum += factor;
	}

	// Normalize and write the result.
	imageStore(tex_out, imageCoord, vec4(t / sum, 0.0, 0.0, 0.0));
}