#include "BilateralFilter.cuh"
#include "LowpassFilter.cuh"
#include "Common.h"
#include <Windows.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cassert>
#include <vector>

#define PI 3.1415926535897932384626433832795

__constant__ float projectionMatrix_3_2, projectionMatrix_2_2;
__constant__ float4 invProjectionMatrix[4];

texture<float, cudaTextureType2D, cudaReadModeElementType> tex_depth;

inline __host__ __device__ float euclideanLength(float a, float b, float sigma)
{
	return expf(-((a - b) * (a - b)) / (2.0f * sigma * sigma));
}

inline __host__ __device__ float gaussian(float x, float y, float sigma)
{
	return expf(-(x * x + y * y) / (2.0f * sigma * sigma)) / (2.0f * PI * sigma * sigma);
}

inline __host__ __device__ float gaussian(float x, float sigma)
{
	return expf(-(x * x) / (2.0f * sigma * sigma)) / (sqrtf(2.0f * PI) * sigma);
}

//@ Backprojection
inline __host__ __device__ float4 backProjection(float P32, float P22, float4 *invP, float4 p_ndc)
{
	float w_c = P32 / (P22 + p_ndc.z);
	float4 p_clip = w_c * p_ndc;
	return invP * p_clip;
}

//@ Backprojection on depth component only.
inline __host__ __device__ float backProjection(float P32, float P22, float z_ndc)
{
	return -P32 / (P22 + z_ndc);
}

//@ Separable bilateral filter.
__global__ void d_GaussianFilter_1D(float *out, int width, int height, int radius, float sigma_s, float sigma_r, bool vertical)
{
	// Calc thread ID(index).
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	if (offset >= width * height){
		return;
	}

	// Get center(current) value.
	float center = tex2D(tex_depth, x, y);
	if (center == 1.0f){
		out[offset] = 1.0f;
		return;
	}
	float4 center_eye = backProjection(projectionMatrix_3_2, projectionMatrix_2_2, invProjectionMatrix,
										make_float4(((float)x / (float)width) * 2.0f - 1.0f,
													((float)y / (float)height) * 2.0f - 1.0f,
													center * 2.0f - 1.0f,
													1.0f));

	float sum = 0.0f;
	float factor = 0.0f;
	float t = 0.0f;
	for (int i = -radius; i <= radius; ++i){
		// Calc sample index.
		int _x, _y;
		if (vertical){
			_x = x;
			_y = y + i;
		}
		else{
			_x = x + i;
			_y = y;
		}

		// Get sample value.
		float sample = tex2D(tex_depth, _x, _y);
		if (sample == 1.0f){
			continue;
		}
		float4 sample_eye = backProjection(projectionMatrix_3_2, projectionMatrix_2_2, invProjectionMatrix,
											make_float4(((float)_x / (float)width) * 2.0f - 1.0f,
														((float)_y / (float)height) * 2.0f - 1.0f,
														sample * 2.0f - 1.0f,
														1.0f));
		float _i;
		if (vertical){
			_i = center_eye.y - sample_eye.y;
		}
		else{
			_i = center_eye.x - sample_eye.x;
		}

		factor = gaussian(_i, sigma_s) * euclideanLength(sample_eye.z, center_eye.z, sigma_r);
		t += factor * sample;
		sum += factor;
	}

	out[offset] = t / sum;
}

BilateralFilter::BilateralFilter(int width, int height,
	unsigned int depthTexID, int bf_radius, float bf_sigma_s, float bf_sigma_r, int bf_nIterations,
	unsigned int thicknessTexID)
	: m_dev_canvas(NULL),
	m_width(width), m_height(height),
	m_grids1D((width * height + 256 - 1) / 256), m_threads1D(256), m_grids2D((width + 16 - 1) / 16, (height + 16 - 1) / 16), m_threads2D(16, 16)
{
	setBFParam(bf_radius, bf_sigma_s, bf_sigma_r, bf_nIterations);

	checkCUDA(cudaMalloc((void **)&m_dev_canvas, sizeof(float) * width * height),
		"Allocating device memory for filter.");

	// Register GL textures to CUDA.
	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[0], depthTexID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone),
		"Registering graphic resources for depth texture.");
	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[1], thicknessTexID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone),
		"Registering graphic resources for thickness texture");

	m_lowpassFilter = LowpassFilter::create(width, height, 20.0f, true);
}

BilateralFilter::~BilateralFilter()
{
	/* This will be needed if cudaDeviceReset() would not be called */
	/*
	checkCUDA(cudaFree(m_dev_canvas), "Deallocating device memory for filter.");

	checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[0]),
		"Unregistering graphic resources for depth texture.");
	checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[1]),
		"Unregistering graphic resources for thickness texture.");

	checkCUDA(cudaFree(m_dev_complex), "Deallocating device memory for cufft");
	checkCUFFT(cufftDestroy(m_plan_fwd), "Deallocating cufft foward plan");
	checkCUFFT(cufftDestroy(m_plan_inv), "Deallocating cufft inverse plan");
	*/

	delete m_lowpassFilter;
}

void
BilateralFilter::run()
{
	// Map device memory between GL and CUDA.
	cudaArray *arr_depth = NULL;
	checkCUDA(cudaGraphicsMapResources(1, m_graphicResources),
		"Mapping graphic resources.");
	checkCUDA(cudaGraphicsSubResourceGetMappedArray(&arr_depth, m_graphicResources[0], 0, 0),
		"Getting mapped cudaArray for depth texture.");

	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	checkCUDA(cudaBindTextureToArray(tex_depth, arr_depth, desc),
		"Binding arr_depth to texture memory.");
	// cudaAddressModeMirror(cudaAddressModeWrap too) does not work, because texture memory bounded to 1D array memory. It only works on 2D array memory.
	//tex_depth.addressMode[0] = cudaAddressModeMirror; 
	//tex_depth.addressMode[1] = cudaAddressModeMirror;

	int radius = m_radius;
	for (int i = 0; i < m_nIterations; ++i){
		// Run filter vertically.
		d_GaussianFilter_1D << < m_grids2D, m_threads2D >> > (m_dev_canvas, m_width, m_height, radius, m_sigma_s, m_sigma_r, true);
		checkCUDA(cudaGetLastError(), "Kernel function error : GaussianFilter_Y.");
		checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, m_dev_canvas, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice),
			"Memcpy to arr_depth(texture memory)");

		// Run filter horizontally.
		d_GaussianFilter_1D << < m_grids2D, m_threads2D >> > (m_dev_canvas, m_width, m_height, radius, m_sigma_s, m_sigma_r, false);
		checkCUDA(cudaGetLastError(), "Kernel function error : GaussianFilter_X.");
		checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, m_dev_canvas, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice),
			"Memcpy to arr_depth(texture memory)");

		// Halve kernel radius.
		radius = (radius >> 1);
	}
	checkCUDA(cudaUnbindTexture(tex_depth), "Unbinding texture memory for arr_depth");
	checkCUDA(cudaGraphicsUnmapResources(1, m_graphicResources),
		"Error : Unmapping graphic resources");
}

void
BilateralFilter::setBFParam(int radius, float sigma_s, float sigma_r, int nIterations)
{
	m_radius = radius;
	m_sigma_s = sigma_s;
	m_sigma_r = sigma_r;
	m_nIterations = nIterations;

	//printf("BFCU r : %d, sigma_s : %.2f, sigma_r : %.2f\n", m_radius, m_sigma_s, m_sigma_r);
}

void
BilateralFilter::setDepthTexture(unsigned int depthTexID)
{
	checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[0]),
		"Error : Unregistering graphic resource for depth texture.");
	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[0], depthTexID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone),
		"Error : Registering graphic resource for depth texture.");
}

void
BilateralFilter::setThicknessTexture(unsigned int thicknessTexID)
{
	checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[1]),
		"Error : Unregistering graphic resource for thickness texture.");
	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[1], thicknessTexID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore),
		"Error : Registering graphic resource for thickness texture.");
}

void
BilateralFilter::filterThickness()
{
	// Filter thickness texture using lowpassfilter.

	// Map device memory between GL and CUDA.
	cudaArray *arr_tex = NULL;
	checkCUDA(cudaGraphicsMapResources(1, &m_graphicResources[1]),
		"Mapping graphic resources.");
	checkCUDA(cudaGraphicsSubResourceGetMappedArray(&arr_tex, m_graphicResources[1], 0, 0),
		"Getting mapped cudaArray for thickness texture.");

	checkCUDA(cudaMemcpyFromArray(m_dev_canvas, arr_tex, 0, 0, m_width * m_height * sizeof(float), cudaMemcpyDeviceToDevice),
		"Memcpy from arr_tex to device memory.");
	
	// Run
	m_lowpassFilter->run(m_dev_canvas);

	// Write to GL texture from CUDA memory.
	checkCUDA(cudaMemcpyToArray(arr_tex, 0, 0, m_dev_canvas, m_width * m_height* sizeof(float), cudaMemcpyDeviceToDevice),
		"Memcpy from device memory to arr_tex");

	// Unmap device memory.
	checkCUDA(cudaGraphicsUnmapResources(1, &m_graphicResources[2]),
		"Error : Unmapping graphic resources");
} 

void BilateralFilter::setMatrix(const glm::mat4 &P, const glm::mat4 &invP)
{
	// Set projection and inverse projectoin matrix to CUDA constant memory.
	checkCUDA(cudaMemcpyToSymbol(projectionMatrix_3_2, &(P[3][2]), sizeof(float)),
		"Memcpy to projmat32 for bilateral filter.");
	checkCUDA(cudaMemcpyToSymbol(projectionMatrix_2_2, &(P[2][2]), sizeof(float)),
		"Memcpy to projmat22 for bilateral filter.");
	checkCUDA(cudaMemcpyToSymbol(invProjectionMatrix, &invP, sizeof(glm::mat4)),
		"Memcpy to invprojmat for bilateral filter.");
}