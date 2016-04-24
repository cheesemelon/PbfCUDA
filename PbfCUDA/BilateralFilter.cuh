#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "glm\glm.hpp"

class LowpassFilter;

//@ Separable bilateral filter using CUDA.
class BilateralFilter{
protected:
	cudaGraphicsResource_t m_graphicResources[2];	//@ To use GL resource in CUDA.
	float *m_dev_canvas;							//@ Temporal memory to write the result of filtering.
	int m_width, m_height;
	int m_grids1D; const int m_threads1D;			//@ # of threads and grids for 1D dispatching.
	dim3 m_grids2D; const dim3 m_threads2D;			//@ # of threads and grids for 2D dispatching.

	int m_radius;			//@ Kernel radius.
	int m_nIterations;		//@ Filter iteration halvaing kernel radius.
	float m_sigma_s;
	float m_sigma_r;

	LowpassFilter *m_lowpassFilter;					//@ Filter to smooth thickness texture.

	BilateralFilter(int width, int height,
		unsigned int depthTexID, int bf_radius, float bf_sigma_s, float bf_sigma_r, int bf_nIterations,
		unsigned int thicknessTexID);
public:
	static BilateralFilter* create(int width, int height,
		unsigned int depthTexID, int bf_radius, float bf_sigma_s, float bf_sigma_r, int bf_nIterations,
		unsigned int thicknessTexID)
	{
		return new BilateralFilter(width, height, depthTexID, bf_radius, bf_sigma_s, bf_sigma_r, bf_nIterations, thicknessTexID);
	}

	~BilateralFilter();

	//@ Run filter. (Bilateral filter only.)
	void run();

	//@ Filter thickness texture.
	void filterThickness();

	void setBFParam(int radius, float sigma_s, float sigma_r, int nIterations);
	void setDepthTexture(unsigned int depthTexID);
	void setThicknessTexture(unsigned int thicknessTexID);

	//@ Set projection and inverse projection matrix. These must be set to use filter.
	void setMatrix(const glm::mat4 &P, const glm::mat4 &invP);
};