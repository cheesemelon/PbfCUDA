#include "BilateralFilter.cuh"
#include "LowpassFilter.cuh"
#include "Common.h"
#include <Windows.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cassert>
#include <vector>

#define PI 3.14159265359f

__constant__ float d_Gaussian[512];

texture<float, cudaTextureType2D, cudaReadModeElementType> tex_depth_original;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_depth;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_dogFiltered;
surface<void, cudaSurfaceType2D> surf_outline;

inline __host__ __device__ float euclideanLength(float a, float b, float sigma){
	return expf(-((a - b) * (a - b)) / (2.0f * sigma * sigma));
}

inline __host__ __device__ float gaussian(float x, float y, float sigma){
	return expf(-(x * x + y * y) / (2.0f * sigma * sigma)) / (2.0f * PI * sigma * sigma);
}

inline __host__ __device__ float gaussian(float x, float sigma){
	return expf(-(x * x) / (2.0f * sigma * sigma)) / (sqrtf(2.0f * PI) * sigma);
}

__constant__ int restorePos;
__constant__ float projectionMatrix_3_2, projectionMatrix_2_2;
//__constant__ float invProjectionMatrix[4];
__constant__ float4 invProjectionMatrix[4];

inline __host__ __device__ float4 mat4Mult(float4 *m, float4 v){
	//float4 Mov0 = make_float4(v.x);
	//float4 Mov1 = make_float4(v.y);
	//float4 Mul0 = m[0] * Mov0;
	//float4 Mul1 = m[1] * Mov1;
	//float4 Add0 = Mul0 + Mul1;
	//float4 Mov2 = make_float4(v.z);
	//float4 Mov3 = make_float4(v.w);
	//float4 Mul2 = m[2] * Mov2;
	//float4 Mul3 = m[3] * Mov3;
	//float4 Add1 = Mul2 + Mul3;
	//return Add0 + Add1;

	return m[0] * make_float4(v.x) + m[1] * make_float4(v.y) + m[2] * make_float4(v.z) + m[3] * make_float4(v.w);
}

inline __host__ __device__ float4 backProjection(float P32, float P22, float4 *invP, float4 p_ndc){
	float w_c = P32 / (P22 + p_ndc.z);
	float4 p_clip = w_c * p_ndc;
	return mat4Mult(invP, p_clip);
}

inline __host__ __device__ float backProjection(float P32, float P22, float z_ndc)
{
	return -P32 / (P22 + z_ndc);
}

__global__ void d_GaussianFilter(float *out, int width, int height, int radius, float sigma_s, float sigma_r){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	if (offset >= width * height){
		return;
	}

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
		for (int j = -radius; j <= radius; ++j){
			float sample = tex2D(tex_depth, x + i, y + j);
			float4 sample_eye = backProjection(projectionMatrix_3_2, projectionMatrix_2_2, invProjectionMatrix,
												make_float4(((float)(x + i) / (float)width) * 2.0f - 1.0f,
															((float)(y + i) / (float)height) * 2.0f - 1.0f,
															sample * 2.0f - 1.0f,
															1.0f));

			//if (restorePos){
				factor = gaussian(center_eye.x - sample_eye.x, center_eye.y - sample_eye.y, sigma_s) * euclideanLength(sample_eye.z, center_eye.z, sigma_r);
			//}
			//else{
			//	factor = gaussian(i, j, sigma_s) * euclideanLength(sample_eye.z, center_eye.z, sigma_r);
			//}

			t += factor * sample;
			sum += factor;
		}
	}

	out[offset] = t / sum;
}
//
//__global__ void ortho(float *out, int width, int height){
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = x + y * width;
//	if (offset >= width * height){
//		return;
//	}
//
//	float center = tex2D(tex_depth, x, y);
//	float4 center_eye = backProjection(make_float4(((float)x / (float)width) * 2.0f - 1.0f,
//													((float)y / (float)height) * 2.0f - 1.0f,
//													center * 2.0f - 1.0f,
//													1.0f));
//
//	float4 frustum = backProjection(make_float4(1, 1, 1, 1));
//
//	float center_ndc_x = center_eye.x / frustum.x;
//	center_ndc_x = (center_ndc_x + 1.0) * 0.5;
//	center_ndc_x = center_ndc_x * (float)width;
//	if (center_ndc_x < 0 || width <= center_ndc_x){
//		return;
//	}
//	float center_ndc_y = center_eye.y / frustum.y;
//	center_ndc_y = (center_ndc_y + 1.0) * 0.5;
//	center_ndc_y = center_ndc_y * (float)height;
//	if (center_ndc_y < 0 || width <= center_ndc_y){
//		return;
//	}
//
//	int _offset = (int)center_ndc_x + (int)center_ndc_y * width;
//	out[_offset] = center;
//}

__global__ void d_GaussianFilter_1D(float *out, float *rweight, int width, int height, int radius, float sigma_s, float sigma_r, bool vertical){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	if (offset >= width * height){
		return;
	}

	float center = tex2D(tex_depth, x, y);
	if (center == 1.0f){
		out[offset] = 1.0f;
		//rweight[offset] = backProjection(make_float4(0, 0, 0, 1)).x;
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
		int _x, _y;
		if (vertical){
			_x = x;
			_y = y + i;
		}
		else{
			_x = x + i;
			_y = y;
		}

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
		//if (restorePos){
			factor = gaussian(_i, sigma_s) * euclideanLength(sample_eye.z, center_eye.z, sigma_r);
		//}
		//else{
		//	factor = gaussian(i, sigma_s) * euclideanLength(sample_eye.z, center_eye.z, sigma_r);
		//	//factor = d_Gaussian[i + radius] * euclideanLength(sample, center, sigma_r);
		//}

		t += factor * sample;
		sum += factor;
	}

	out[offset] = t / sum;
	//rweight[offset] = viewFrustumWidth;
}

__global__ void d_DOGFilter(float *out, int width, int height, int radius, float sigma1, float sigma2){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	if (offset >= width * height){
		return;
	}

	// Cross filtering
	float sum = 0.0f;
	for (int i = -radius; i <= radius; ++i){
		sum += d_Gaussian[i + radius] * tex2D(tex_depth, x + i, y);
		sum += d_Gaussian[i + radius] * tex2D(tex_depth, x, y + i);
	}
	out[offset] = sum / ((2 * radius + 1) + (2 * radius));

	// Normal filtering
	/*
	float sum = 0.0f;
	for (int i = -radius; i <= radius; ++i){
		for (int j = -radius; j <= radius; ++j){
			sum += (gaussian(i, j, sigma1) - gaussian(i, j, sigma2)) * tex2D(tex_depth, x + i, y + i);
		}
	}
	out[offset] = sum / ((2 * radius + 1) * (2 * radius + 1));
	*/

	// Box filtering
	/*
	float sum = 0.0f;
	for (int i = -1; i <= 1; ++i){
		for (int j = -1; j <= 1; ++j){
			sum += (gaussian(i, j, sigma1) - gaussian(i, j, sigma2)) * tex2D(tex_depth, x + i, y + i);
		}
	}
	out[offset] = sum / 9.0f;
	*/
}

__global__ void d_detectZeroCrossing(int width, int height){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	if (offset >= width * height){
		return;
	}

	float center = tex2D(tex_dogFiltered, x, y);
	unsigned char res = 0x00;
	if (center < 0){
		surf2Dwrite(res, surf_outline, x, y, cudaBoundaryModeClamp);
		return;
	}
	float maxDiff = 0.0f;
	for (int i = -1; i <= 1; ++i){
		float sampleX = tex2D(tex_dogFiltered, x + i, y);
		if (center * sampleX < 0){
			float diff = fabs(center) + fabs(sampleX);
			if (diff > maxDiff){
				maxDiff = diff;
			}
		}

		float sampleY = tex2D(tex_dogFiltered, x, y + i);
		if (center * sampleY < 0){
			float diff = fabs(center) + fabs(sampleY);
			if (diff > maxDiff){
				maxDiff = diff;
			}
		}
	}

	if (maxDiff > 0.000001f){
	//if (maxDiff > 0.0f){
		res = 0xff;
	}
	surf2Dwrite(res, surf_outline, x, y, cudaBoundaryModeClamp);
}

//__global__ void d_findMaxValue(float *out, int width, int height){
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = x + y * width;
//	if (offset > width * height){
//		return;
//	}
//
//	__shared__ float sdata[16 * 16];
//	unsigned int tid = threadIdx.x + blockDim.x * threadIdx.y;
//	//sdata[tid] = tex2D(tex_depth, x, y);
//	sdata[tid] = getEyePos(tex2D(tex_depth, x, y));
//	__syncthreads();
//
//	for (unsigned int s = 1; s < blockDim.x * blockDim.y; s *= 2){
//		if (tid % (2 * s) == 0){
//			//float _max = fmaxf(sdata[tid], sdata[tid + s]);
//			//if (_max == 1.0f){
//			//	sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
//			//}
//			//else{
//			//	sdata[tid] = _max;
//			//}
//
//			sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
//		}
//		__syncthreads();
//	}
//
//	out[blockIdx.x + gridDim.x * blockIdx.y] = sdata[0];
//}
//
//__global__ void d_findMinValue(float *out, int width, int height){
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = x + y * width;
//	if (offset > width * height){
//		return;
//	}
//
//	__shared__ float sdata[16 * 16];
//	unsigned int tid = threadIdx.x + blockDim.x * threadIdx.y;
//	//sdata[tid] = tex2D(tex_depth, x, y);
//	sdata[tid] = getEyePos(tex2D(tex_depth, x, y));
//	__syncthreads();
//
//	for (unsigned int s = 1; s < blockDim.x * blockDim.y; s *= 2){
//		if (tid % (2 * s) == 0){
//			sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
//		}
//		__syncthreads();
//	}
//
//	out[blockIdx.x + gridDim.x * blockIdx.y] = sdata[0];
//}

__global__ void d_scaling(float *data, float min, float max, int bound){
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset > bound){
		return;
	}

	data[offset] = (data[offset] - min) / (max - min);
}

__global__ void test(float *canvas, int width, int height){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	if (offset > width * height){
		return;
	}

	if (offset < 2048){
		canvas[offset] = 0.0f;
	}
	else{
		canvas[offset] = tex2D(tex_depth, x, y);
	}
}

__global__ void avg(float *canvas1, float *canvas2, int width, int height){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	if (offset > width * height){
		return;
	}

	canvas1[offset] = (canvas1[offset] + canvas2[offset]) / 2;
}

BilateralFilter::BilateralFilter(int width, int height,
	GLuint depthTexID, int bf_radius, int bf_sigma_s, float bf_sigma_r, int bf_nIterations,
	GLuint outlineTexID, int dog_radius, int dog_sigma, float dog_similarity,
	GLuint thicknessTexID)
	: m_dev_canvas(NULL),
	m_width(width), m_height(height),
	m_grids1D((width * height + 256 - 1) / 256), m_threads1D(256), m_grids2D((width + 16 - 1) / 16, (height + 16 - 1) / 16), m_threads2D(16, 16),
	m_lowpassFilter(NULL)
{
	setBFParam(bf_radius, bf_sigma_s, bf_sigma_r, bf_nIterations);
	setDOGParam(dog_radius, dog_sigma, dog_similarity);

	checkCUDA(cudaMalloc((void **)&m_dev_canvas, sizeof(float) * width * height),
		"Allocating device memory for filter.");

	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[0], depthTexID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone),
		"Registering graphic resources for depth texture.");
	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[1], outlineTexID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore),
		"Registering graphic resources for outline texture");
	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[2], thicknessTexID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone),
		"Registering graphic resources for thickness texture");

	m_lowpassFilter = LowpassFilter::create(width, height, 20.0f, true);
}

BilateralFilter::~BilateralFilter() {
	/* This will be needed if cudaDeviceReset() would not be called */
	/*
	checkCUDA(cudaFree(m_dev_canvas), "Deallocating device memory for filter.");

	checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[0]),
		"Unregistering graphic resources for depth texture.");
	checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[1]),
		"Unregistering graphic resources for outline texture.");
	checkCUDA(cudaGraphicsUnregisterResource(resource[2]),
		"Unregistering graphic resources for thickness texture.");

	checkCUDA(cudaFree(m_dev_complex), "Deallocating device memory for cufft");
	checkCUFFT(cufftDestroy(m_plan_fwd), "Deallocating cufft foward plan");
	checkCUFFT(cufftDestroy(m_plan_inv), "Deallocating cufft inverse plan");
	*/

	delete m_lowpassFilter;
}

void
BilateralFilter::filter(bool renderOutline){
	cudaArray *arr_depth = NULL;
	cudaArray *arr_outline = NULL;
	checkCUDA(cudaGraphicsMapResources(2, m_graphicResources),
		"Mapping graphic resources.");
	checkCUDA(cudaGraphicsSubResourceGetMappedArray(&arr_depth, m_graphicResources[0], 0, 0),
		"Getting mapped cudaArray for depth texture.");
	checkCUDA(cudaGraphicsSubResourceGetMappedArray(&arr_outline, m_graphicResources[1], 0, 0),
		"Getting mapped cudaArray for outline texture.");

	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	checkCUDA(cudaBindTextureToArray(tex_depth, arr_depth, desc),
		"Binding arr_depth to texture memory.");
	tex_depth.addressMode[0] = cudaAddressModeMirror;
	tex_depth.addressMode[1] = cudaAddressModeMirror;

	float *depth_original;
	cudaMalloc((void **)&depth_original, sizeof(float) * m_width * m_height);
	cudaMemcpyFromArray(depth_original, arr_depth, 0, 0, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice);
	cudaBindTexture2D(0, tex_depth_original, depth_original, desc, m_width, m_height, sizeof(float) * m_width);

	cudaMemcpyToSymbol(restorePos, &m_restorePos, sizeof(int));
	//std::cout << m_intensityRange << " " << m_bf_sigma_r << std::endl;

	//test << < m_grids2D, m_threads2D >> > (m_dev_canvas, m_width, m_height);
	//checkCUDA(cudaGetLastError());
	//checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, m_dev_canvas, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));

	//FIXME TEMP
	//float *dev_max;
	//cudaMalloc(&dev_max, m_grids2D.x * m_grids2D.y * sizeof(float));
	//cudaMemset(dev_max, 0, m_grids2D.x * m_grids2D.y * sizeof(float));
	//d_findMaxValue << <m_grids2D, m_threads2D >> >(dev_max, m_width, m_height);
	//std::vector<float> max(m_grids2D.x * m_grids2D.y);
	//cudaMemcpy(max.data(), dev_max, m_grids2D.x * m_grids2D.y * sizeof(float), cudaMemcpyDeviceToHost);
	//float maxv = -99999.0f;
	//for (auto &v : max){
	//	if (maxv < v){
	//		maxv = v;
	//	}
	//}
	//d_findMinValue << < m_grids2D, m_threads2D >> > (dev_max, m_width, m_height);
	//cudaMemcpy(max.data(), dev_max, m_grids2D.x * m_grids2D.y * sizeof(float), cudaMemcpyDeviceToHost);
	//float minv = 999999.0f;
	//for (auto &v : max){
	//	if (minv > v){
	//		minv = v;
	//	}
	//}
	//std::cout << maxv << " " << minv << std::endl;

	if (renderOutline){
		checkCUDA(cudaBindTexture2D(0, tex_dogFiltered, m_dev_canvas, desc, m_width, m_height, sizeof(float) * m_width),
			"Binding dog filtered data to texture memory.");
		tex_dogFiltered.addressMode[0] = cudaAddressModeMirror;
		tex_dogFiltered.addressMode[1] = cudaAddressModeMirror;

		desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		checkCUDA(cudaBindSurfaceToArray(surf_outline, arr_outline, desc),
			"Binding arr_outline to surface memory.");

		checkCUDA(cudaMemcpyToSymbol(d_Gaussian, m_dog_Gaussian.data(), sizeof(float) * m_dog_Gaussian.size()),
			"Memcpy to Gaussian symbol for dog filter.");
		d_DOGFilter << < m_grids2D, m_threads2D >> > (m_dev_canvas, m_width, m_height, m_dog_radius, static_cast<float>(m_dog_sigma), m_dog_similarity * static_cast<float>(m_dog_sigma));
		checkCUDA(cudaGetLastError(), "Kernel function error : DOGFilter.");
		d_detectZeroCrossing << < m_grids2D, m_threads2D >> > (m_width, m_height);
		checkCUDA(cudaGetLastError(), "Kernel function error : ExtractZeroCrossing.");

		checkCUDA(cudaUnbindTexture(tex_dogFiltered), "Unbinding texture memory for dog filtered data.");
	}

	//cudaMemcpyFromArray(m_dev_canvas, arr_depth, 0, 0, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice);
	//d_scaling <<< m_grids1D, m_threads1D>>>(m_dev_canvas, minv, maxv, m_width * m_height);
	//cudaMemcpyToArray(arr_depth, 0, 0, m_dev_canvas, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice);



	//std::cout << "RESTORE POS : " << m_restorePos << " sigma r : " << m_bf_sigma_r << std::endl;

	float *dev_rweight;
	cudaMalloc(&dev_rweight, m_width * m_height * sizeof(float));
	
	for (auto &Gaussian : m_bf_Gaussians){
		int radius = (Gaussian.size() >> 1);
		//checkCUDA(cudaMemcpyToSymbol(d_Gaussian, Gaussian.data(), sizeof(float) * Gaussian.size()),
		//	"Memcpy to Gaussian symbol for bilateral filter.");

		d_GaussianFilter_1D << < m_grids2D, m_threads2D >> > (m_dev_canvas, dev_rweight, m_width, m_height, radius, m_bf_sigma_s, m_bf_sigma_r, true);
		checkCUDA(cudaGetLastError(), "Kernel function error : GaussianFilter_Y.");
		checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, m_dev_canvas, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice),
			"Memcpy to arr_depth(texture memory)");

		d_GaussianFilter_1D << < m_grids2D, m_threads2D >> > (m_dev_canvas, dev_rweight, m_width, m_height, radius, m_bf_sigma_s, m_bf_sigma_r, false);
		checkCUDA(cudaGetLastError(), "Kernel function error : GaussianFilter_X.");
		checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, m_dev_canvas, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice),
			"Memcpy to arr_depth(texture memory)");
	}

	//if (m_restorePos){
	//	checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, depth_original, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice),
	//		"Memcpy to arr_depth(texture memory)");

	//	for (auto &Gaussian : m_bf_Gaussians){
	//		int radius = (Gaussian.size() >> 1);

	//		d_GaussianFilter_1D << < m_grids2D, m_threads2D >> > (dev_rweight, dev_rweight, m_width, m_height, radius, m_bf_sigma_s, m_bf_sigma_r, false);
	//		checkCUDA(cudaGetLastError(), "Kernel function error : GaussianFilter_Y.");
	//		checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, dev_rweight, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice),
	//			"Memcpy to arr_depth(texture memory)");

	//		d_GaussianFilter_1D << < m_grids2D, m_threads2D >> > (dev_rweight, dev_rweight, m_width, m_height, radius, m_bf_sigma_s, m_bf_sigma_r, true);
	//		checkCUDA(cudaGetLastError(), "Kernel function error : GaussianFilter_X.");
	//		checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, dev_rweight, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice),
	//			"Memcpy to arr_depth(texture memory)");
	//	}

	//	avg << < m_grids2D, m_threads2D >> > (m_dev_canvas, dev_rweight, m_width, m_height);
	//	checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, m_dev_canvas, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice),
	//		"Memcpy to arr_depth(texture memory)");
	//}

	//if (m_restorePos){
	//	d_GaussianFilter << < m_grids2D, m_threads2D >> > (m_dev_canvas, m_width, m_height, 20, m_bf_sigma_s, m_bf_sigma_r);
	//	checkCUDA(cudaGetLastError(), "Kernel function error : GaussianFilter_X.");
	//	checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, m_dev_canvas, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice),
	//		"Memcpy to arr_depth(texture memory)");
	//}

	//cudaMemcpy(m_dev_canvas, std::vector<float>(m_width * m_height, 1.0f).data(), sizeof(float) *m_width * m_height, cudaMemcpyHostToDevice);
	//ortho << < m_grids2D, m_threads2D >> > (m_dev_canvas, m_width, m_height);
	//	checkCUDA(cudaGetLastError(), "Kernel function error : GaussianFilter_Y.");
	//	checkCUDA(cudaMemcpyToArray(arr_depth, 0, 0, m_dev_canvas, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice),
	//		"Memcpy to arr_depth(texture memory)");
	checkCUDA(cudaUnbindTexture(tex_depth), "Unbinding texture memory for arr_depth");
	
	//std::vector<float> temp(m_width * m_height);
	//cudaMemcpy(temp.data(), dev_rweight, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToHost);
	//float min = 0, max = 0;
	//for (int i = 0; i < m_width * m_height; ++i){
	//	if (temp[i] < min){
	//		min = temp[i];
	//	}
	//	if (temp[i] > max){
	//		max = temp[i];
	//	}
	//}
	//cout << "min : " << min << ", max : " << max << endl;
	cudaFree(dev_rweight);

	cudaUnbindTexture(tex_depth_original);
	cudaFree(depth_original);

	checkCUDA(cudaGraphicsUnmapResources(2, m_graphicResources),
		"Error : Unmapping graphic resources");
}

void
BilateralFilter::setBFParam(int radius, int sigma_s, float sigma_r, int nIterations) {
	m_bf_radius = radius;
	m_bf_sigma_s = sigma_s;
	m_bf_sigma_r = sigma_r;
	m_bf_nIterations = nIterations;

	m_bf_Gaussians.clear();
	for (int i = 0; i < nIterations; ++i){
		std::vector<float> Gaussian;
		for (int i = 0; i < 2 * radius + 1; ++i){
			Gaussian.push_back(gaussian(i - radius, sigma_s));
		}
		m_bf_Gaussians.push_back(Gaussian);
		radius >>= 1;
		if (radius == 0){
			break;
		}
	}
}

void
BilateralFilter::setDOGParam(int radius, int sigma, float similarity){
	m_dog_radius = radius;
	m_dog_sigma = sigma;
	m_dog_similarity = similarity;

	m_dog_Gaussian.clear();
	for (int i = 0; i < 2 * radius + 1; ++i){
		m_dog_Gaussian.push_back(gaussian(i - radius, sigma) - gaussian(i - radius, similarity * sigma));
	}
}

void
BilateralFilter::setDepthTexture(GLuint depthTexID){
	checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[0]),
		"Error : Unregistering graphic resource for depth texture.");
	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[0], depthTexID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone),
		"Error : Registering graphic resource for depth texture.");
}

void
BilateralFilter::setOutlineTexture(GLuint outlineTexID){
	checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[1]),
		"Error : Unregistering graphic resource for outline texture.");
	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[1], outlineTexID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore),
		"Error : Registering graphic resource for outline texture.");
}

void
BilateralFilter::setThicknessTexture(GLuint thicknessTexID) {
	checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[2]),
		"Error : Unregistering graphic resource for thickness texture.");
	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[2], thicknessTexID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore),
		"Error : Registering graphic resource for thickness texture.");
}

void
BilateralFilter::filterThickness() {
	cudaArray *arr_tex = NULL;
	checkCUDA(cudaGraphicsMapResources(1, &m_graphicResources[2]),
		"Mapping graphic resources.");
	checkCUDA(cudaGraphicsSubResourceGetMappedArray(&arr_tex, m_graphicResources[2], 0, 0),
		"Getting mapped cudaArray for thickness texture.");

	checkCUDA(cudaMemcpyFromArray(m_dev_canvas, arr_tex, 0, 0, m_width * m_height * sizeof(float), cudaMemcpyDeviceToDevice),
		"Memcpy from arr_tex to device memory.");
	
	m_lowpassFilter->run(m_dev_canvas);

	checkCUDA(cudaMemcpyToArray(arr_tex, 0, 0, m_dev_canvas, m_width * m_height* sizeof(float), cudaMemcpyDeviceToDevice),
		"Memcpy from device memory to arr_tex");

	checkCUDA(cudaGraphicsUnmapResources(1, &m_graphicResources[2]),
		"Error : Unmapping graphic resources");
} 

void BilateralFilter::setProjectionMatrix(glm::mat4 matrix){
	checkCUDA(cudaMemcpyToSymbol(projectionMatrix_3_2, &(matrix[3][2]), sizeof(float)),
		"Memcpy to projmat32 for bilateral filter.");
	checkCUDA(cudaMemcpyToSymbol(projectionMatrix_2_2, &(matrix[2][2]), sizeof(float)),
		"Memcpy to projmat22 for bilateral filter.");

	glm::mat4 inv = glm::inverse(matrix);
	checkCUDA(cudaMemcpyToSymbol(invProjectionMatrix, &inv, sizeof(glm::mat4)),
		"Memcpy to invprojmat for bilateral filter.");

	m_P = matrix;
}