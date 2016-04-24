#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

class LowpassFilter;

//@ nCk. This is implemented as recursive function.
extern int nchoosek(int n, int k);

//@ Shiftable bilateral filter. (Trigonometric function bilateral filter)
class ShiftableBilateralFilter{
protected:
	cudaGraphicsResource_t m_graphicResources[1];				//@ To use GL resource in CUDA.
	float *m_dev_data;
	float *m_dev_out1, *m_dev_out2;
	float *m_dev_phi1, *m_dev_phi2, *m_dev_phi3, *m_dev_phi4;
	int m_width, m_height;
	int m_N, m_M;
	float m_perNsquare, m_gamma;
	int m_grids1D; const int m_threads1D;						//@ # of threads and grids for 1D dispatching.
	dim3 m_grids2D; const dim3 m_threads2D;						//@ # of threads and grids for 2D dispatching.

	LowpassFilter *m_lowpassFilter;								//@ Spatial filter.
public:
	ShiftableBilateralFilter() :
		m_dev_data(NULL), m_dev_out1(NULL), m_dev_out2(NULL),
		m_dev_phi1(NULL), m_dev_phi2(NULL), m_dev_phi3(NULL), m_dev_phi4(NULL),
		m_width(0), m_height(0), m_N(0), m_M(0), m_perNsquare(0.0f), m_gamma(0.0f),
		m_grids1D(0), m_threads1D(256), m_grids2D(0, 0), m_threads2D(16, 16),
		m_lowpassFilter(NULL)
	{
		m_graphicResources[0] = 0;
	}

	~ShiftableBilateralFilter();

	//@ Init filter. This must be done before to use filter.
	void	init(int width, int height, float radius, int sigma_r, float tolerance, unsigned int textureID);

	//@ Run filter.
	void	run();

	void	setRadius(float radius);
	void	setParam(int sigma_r, float tolerance);

	//@ Get filter iteration. Spatial filter will be run 4 times of iteration.
	int		getNumIterations()		{ return m_N - 2 * m_M; }
};