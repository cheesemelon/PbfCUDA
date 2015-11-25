#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

class LowpassFilter{
protected:
	cufftComplex *m_dev_complex;
	cufftHandle m_plan_fwd, m_plan_inv;
	int m_width, m_height;
	float m_radius;
	bool m_normalize;
	const dim3 m_grids2D; const dim3 m_threads2D;

	LowpassFilter(int width, int height, float radius, bool normalize);

public:
	static LowpassFilter* create(int width, int height, float radius, bool normalize = false){
		return new LowpassFilter(width, height, radius, normalize);
	}

	~LowpassFilter();

	void run(float *dev_data);
	
	// FIXME
	//void run(GLuint textureID);

	void setRadius(float radius) { m_radius = radius; }
};