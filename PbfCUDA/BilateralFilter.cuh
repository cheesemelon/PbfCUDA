#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include "temp.h"
#include "glm\glm.hpp"

class LowpassFilter;

class BilateralFilter{
protected:
	cudaGraphicsResource_t m_graphicResources[3];
	float *m_dev_canvas;
	int m_width, m_height;
	int m_grids1D; const int m_threads1D;
	dim3 m_grids2D; const dim3 m_threads2D;

	int m_bf_radius, m_bf_nIterations;
	float m_bf_sigma_s;
	float m_bf_sigma_r;

	int m_dog_radius, m_dog_sigma;
	float m_dog_similarity;
	std::vector<float> m_dog_Gaussian;

	LowpassFilter *m_lowpassFilter;

	BilateralFilter(int width, int height,
		GLuint depthTexID, int bf_radius, float bf_sigma_s, float bf_sigma_r, int bf_nIterations,
		GLuint outlineTexID, int dog_radius, int dog_sigma, float dog_similarity,
		GLuint thicknessTexID);
public:
	static BilateralFilter* create(int width, int height,
		GLuint depthTexID, int bf_radius, float bf_sigma_s, float bf_sigma_r, int bf_nIterations,
		GLuint outlineTexID, int dog_radius, int dog_sigma, float dog_similarity,
		GLuint thicknessTexID)
	{
		return new BilateralFilter(width, height, depthTexID, bf_radius, bf_sigma_s, bf_sigma_r, bf_nIterations, outlineTexID, dog_radius, dog_sigma, dog_similarity, thicknessTexID);
	}

	~BilateralFilter();

	void filter(bool renderOutline);
	void setBFParam(int radius, float sigma_s, float sigma_r, int nIterations);
	void setDOGParam(int radius, int sigma, float similarity);
	void setDepthTexture(GLuint depthTexID);
	void setOutlineTexture(GLuint outlineTexID);
	void setThicknessTexture(GLuint thicknessTexID);
	void setMatrix(const glm::mat4 &P, const glm::mat4 &invP);
	void filterThickness();
};