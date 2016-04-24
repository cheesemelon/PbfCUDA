#include "SFBF.cuh"
#include "LowpassFilter.cuh"
#include "gl\glew.h"
#include <cuda_gl_interop.h>
#include "Common.h"

int nchoosek(int n, int k){
	if (k == 0){
		return 1;
	}
	else{
		return (n * nchoosek(n - 1, k - 1)) / k;
	}
}

//@ Change intensity range from [0.0 1.0] to [0.0 255.0].
__global__ void d_scale_up(float *data, int bound){
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset < bound){
		data[offset] *= 255.0f;
	}
}

__global__ void d_bilateral_calculate_phis(const float *in, float *phi1, float *phi2, float *phi3, float *phi4, float coeff, int bound){
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset < bound){
		float _in = in[offset];
		float temp1 = cos(coeff * _in);
		float temp2 = sin(coeff * _in);

		phi1[offset] = temp1 * _in;
		phi2[offset] = temp2 * _in;
		phi3[offset] = temp1;
		phi4[offset] = temp2;
	}
}

__global__ void d_bilateral_summation(float *out1, float *out2, const float *in, float *phi1, float *phi2, float *phi3, float *phi4, float trigonalCoeff, float seriesCoeff, int bound){
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset < bound){
		float _in = in[offset];
		float temp1 = cos(trigonalCoeff * _in);
		float temp2 = sin(trigonalCoeff * _in);

		out1[offset] += seriesCoeff * (temp1 * phi1[offset] + temp2 * phi2[offset]);
		out2[offset] += seriesCoeff * (temp1 * phi3[offset] + temp2 * phi4[offset]);
	}
}

//@ Normalize the result and change intensity range from [0.0 255.0] to [0.0 1.0].
__global__ void d_bilateral_final(float *in, float *out1, float *out2, int bound){
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset < bound){
		float _out2 = out2[offset];
		//if (_out2 > 0.0001f){
		if (in[offset] != 255.0f && _out2 > 0.0001f){
			in[offset] = (out1[offset] / _out2) * 0.003921568627f; // /255.0f
		}
		else{
			in[offset] *= 0.003921568627f; // *=255.0f;
		}
	}
}

ShiftableBilateralFilter::~ShiftableBilateralFilter() {
	if (m_dev_data != NULL)		checkCUDA(cudaFree(m_dev_data), "Error : Deallocating device memory for filter");
	if (m_dev_out1 != NULL)		checkCUDA(cudaFree(m_dev_out1), "Error : Deallocating device memory for filter");
	if (m_dev_out2 != NULL)		checkCUDA(cudaFree(m_dev_out2), "Error : Deallocating device memory for filter");
	if (m_dev_phi1 != NULL)		checkCUDA(cudaFree(m_dev_phi1), "Error : Deallocating device memory for filter");
	if (m_dev_phi2 != NULL)		checkCUDA(cudaFree(m_dev_phi2), "Error : Deallocating device memory for filter");
	if (m_dev_phi3 != NULL)		checkCUDA(cudaFree(m_dev_phi3), "Error : Deallocating device memory for filter");
	if (m_dev_phi4 != NULL)		checkCUDA(cudaFree(m_dev_phi4), "Error : Deallocating device memory for filter");

	delete m_lowpassFilter;

	checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[0]),
		"Error : Unregistering graphic resources");
}

void
ShiftableBilateralFilter::init(int width, int height, float radius, int sigma_r, float tolerance, unsigned int textureID) {
	m_width = width;
	m_height = height;
	setParam(sigma_r, tolerance);
	m_grids1D = (width * height + 256 - 1) / 256;
	m_grids2D = dim3((width + 16 - 1) / 16, (height + 16 - 1) / 16);

	delete m_lowpassFilter;
	m_lowpassFilter = LowpassFilter::create(width, height, radius);

	/* This will be needed if cudaDeviceReset() would not be called */
	//if (m_dev_data != NULL)		res_cuda = cudaFree(m_dev_data);
	//if (m_dev_out1 != NULL)		res_cuda = cudaFree(m_dev_out1);
	//if (m_dev_out2 != NULL)		res_cuda = cudaFree(m_dev_out2);
	//if (m_dev_phi1 != NULL)		res_cuda = cudaFree(m_dev_phi1);
	//if (m_dev_phi2 != NULL)		res_cuda = cudaFree(m_dev_phi2);
	//if (m_dev_phi3 != NULL)		res_cuda = cudaFree(m_dev_phi3);
	//if (m_dev_phi4 != NULL)		res_cuda = cudaFree(m_dev_phi4);
	//if (m_dev_complex != NULL)	res_cuda = cudaFree(m_dev_complex);
	//if (m_plan_fwd != 0)		res_cufft = cufftDestroy(m_plan_fwd);
	//if (m_plan_inv != 0)		res_cufft = cufftDestroy(m_plan_inv);
	//if (res_cuda != CUDA_SUCCESS || res_cufft != CUFFT_SUCCESS){
	//	std::cout << "Error releasing filter resources" << std::endl;
	//}
	//checkCUDA(cudaGraphicsUnregisterResource(m_graphicResources[0]),
	//	"Error : Unregistering graphic resources");

	checkCUDA(cudaMalloc((void **)&m_dev_data, sizeof(float) * width * height),
		"Error : Allocating device memory for filter");
	checkCUDA(cudaMalloc((void **)&m_dev_out1, sizeof(float) * width * height),
		"Error : Allocating device memory for filter");
	checkCUDA(cudaMalloc((void **)&m_dev_out2, sizeof(float) * width * height),
		"Error : Allocating device memory for filter");
	checkCUDA(cudaMalloc((void **)&m_dev_phi1, sizeof(float) * width * height),
		"Error : Allocating device memory for filter");
	checkCUDA(cudaMalloc((void **)&m_dev_phi2, sizeof(float) * width * height),
		"Error : Allocating device memory for filter");
	checkCUDA(cudaMalloc((void **)&m_dev_phi3, sizeof(float) * width * height),
		"Error : Allocating device memory for filter");
	checkCUDA(cudaMalloc((void **)&m_dev_phi4, sizeof(float) * width * height),
		"Error : Allocating device memory for filter");

	// Register GL textures to CUDA.
	checkCUDA(cudaGraphicsGLRegisterImage(&m_graphicResources[0], textureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone),
		"Error : Registering graphic resources");
}

void
ShiftableBilateralFilter::run() {
	cudaArray *arr_data = NULL;

	// Map device memory between GL and CUDA.
	checkCUDA(cudaGraphicsMapResources(1, m_graphicResources),
		"Error : Mapping graphic resources");
	checkCUDA(cudaGraphicsSubResourceGetMappedArray(&arr_data, m_graphicResources[0], 0, 0),
		"Error : Getting mapped cudaArray");
	checkCUDA(cudaMemcpyFromArray(m_dev_data, arr_data, 0, 0, m_width * m_height * sizeof(float), cudaMemcpyDeviceToDevice),
		"Error : Memcpy from cudaArray");
	d_scale_up << <m_grids1D, m_threads1D >> > (m_dev_data, m_width * m_height);
	checkCUDA(cudaGetLastError(), "scale up kernel function");

	checkCUDA(cudaMemset(m_dev_out1, 0, sizeof(float) * m_width * m_height),
		"Error : Initializing device memory for filter");
	checkCUDA(cudaMemset(m_dev_out2, 0, sizeof(float) * m_width * m_height),
		"Error : Initializing device memory for filter");

	// Run filter using trigonometric range kernel.
	int bound = m_width * m_height;
	for (int k = m_M; k <= m_N - m_M; ++k){
		float trigonalCoeff = (float)(2 * k - m_N) * m_gamma;
		float seriesCoeff = nchoosek(m_N, k) * m_perNsquare;

		d_bilateral_calculate_phis << < m_grids1D, m_threads1D >> >(
			m_dev_data,
			m_dev_phi1, m_dev_phi2, m_dev_phi3, m_dev_phi4,
			trigonalCoeff, bound);

		// Run spatial filter.
		m_lowpassFilter->run(m_dev_phi1);
		m_lowpassFilter->run(m_dev_phi2);
		m_lowpassFilter->run(m_dev_phi3);
		m_lowpassFilter->run(m_dev_phi4);

		d_bilateral_summation << < m_grids1D, m_threads1D >> >(
			m_dev_out1, m_dev_out2,
			m_dev_data,
			m_dev_phi1, m_dev_phi2, m_dev_phi3, m_dev_phi4,
			trigonalCoeff, seriesCoeff, bound);
	}
	d_bilateral_final << < m_grids1D, m_threads1D >> >(
		m_dev_data,
		m_dev_out1, m_dev_out2,
		bound);
	checkCUDA(cudaGetLastError(),
		"Error : bilateral kernel function");

	// Write the result and unmap device memory.
	checkCUDA(cudaMemcpyToArray(arr_data, 0, 0, m_dev_data, m_width * m_height * sizeof(float), cudaMemcpyDeviceToDevice),
		"Error : Memcpy to cudaArray");
	checkCUDA(cudaGraphicsUnmapResources(1, m_graphicResources),
		"Error : Unmapping graphic resources");
}

void
ShiftableBilateralFilter::setRadius(float radius) {
	m_lowpassFilter->setRadius(radius);
}

void
ShiftableBilateralFilter::setParam(int sigma_r, float tolerance) {
	m_N = (int)ceilf(0.405f * 65025.0f / static_cast<float>(sigma_r * sigma_r));
	m_perNsquare = 1.0f / static_cast<float>(m_N * m_N);
	m_gamma = 1.0f / (sqrtf(static_cast<float>(m_N)) * static_cast<float>(sigma_r));
	if (tolerance == 0.0f){
		m_M = 0;
	}
	else{
		if (sigma_r > 40){
			m_M = 0;
		}
		else if (sigma_r > 10){
			float sumCoeffs = 0.0f;
			for (int k = 0; k <= m_N / 2; ++k){
				sumCoeffs += static_cast<float>(nchoosek(m_N, k)) * m_perNsquare;
				if (sumCoeffs > tolerance * 0.5f){
					m_M = k;
					break;
				}
			}
		}
		else{
			m_M = static_cast<int>(ceilf(0.5f * (static_cast<float>(m_N)-sqrtf(4.0f * static_cast<float>(m_N)* logf(2.0f / tolerance)))));
		}
	}
}