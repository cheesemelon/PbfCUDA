#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

class LowpassFilter{
protected:
	cufftComplex *m_dev_complex;		//@ Device memory for cufft.
	cufftHandle m_plan_fwd, m_plan_inv;	//@ FFT plans for forward and inverse transform.
	int m_width, m_height;
	float m_radius;						//@ Frequency radius to passthrough.
	bool m_normalize;
	const dim3 m_grids2D; const dim3 m_threads2D;

	cudaGraphicsResource_t m_graphicResource;
	float *m_dev_data;

	LowpassFilter(int width, int height, float radius, bool normalize);

public:
	static LowpassFilter* create(int width, int height, float radius, bool normalize = false){
		return new LowpassFilter(width, height, radius, normalize);
	}

	~LowpassFilter();

	//@ Run on CUDA device data.
	void run(float *dev_data);
	
	//@ Map GL resource to CUDA.
	void setTexture(unsigned int textureID);

	//@ Run on GL texture. It needs to be set texture.
	void run();

	void setRadius(float radius)	{ m_radius = radius; }
};