#pragma once

#include "gl\glew.h"
#include "glm\glm.hpp"
#include "Shader.h"
#include "Texture.h"

//@ Separable bilateral filter using compute shader.
class BilateralFilterGL{
protected:
	int m_width, m_height;
	int m_radius;				//@ Kernel radius.
	float m_sigma_s, m_sigma_r;
	int m_nIterations;			//@ Filter iteration halving kernel radius.
	Shader *m_shader;
	Texture *m_tex_canvas;		//@ Temporal memory to write the result of filtering.

	BilateralFilterGL(int width, int height,
					int radius, float sigma_s, float sigma_r, int nIterations,
					const glm::mat4 &P, const glm::mat4 &invP);

public:
	static BilateralFilterGL* create(int width, int height,
									int radius, float sigma_s, float sigma_r, int nIterations,
									const glm::mat4 &P, const glm::mat4 &invP)
	{
		return new BilateralFilterGL(width, height, radius, sigma_s, sigma_r, nIterations, P, invP);
	}

	~BilateralFilterGL()			{ delete m_shader, m_tex_canvas; }

	//@ Setter of filter parameters.
	void	setParameters(int radius, float sigma_s, float sigma_r, int nIterations);

	//@ Run filter on depthTextureID(2D single channel texture).
	void	run(GLuint depthTextureID);

	// Setters
	//
	void	setRadius(int radius)		{ m_radius = radius; }
	void	setSigmaS(float sigma_s)	{ m_sigma_s = sigma_s; }
	void	setSigmaR(float sigma_r)	{ m_sigma_r = sigma_r; }
	void	setIteration(int iteration) { m_nIterations = iteration; }

	// For on/off depth correction, 3D filtering.
	//
	bool	bDepthCorrection;
	bool	b3DFiltering;

	//@ Toggle depth correction. Default is true.
	void	toggleDepthCorrection()
	{
		bDepthCorrection = !bDepthCorrection;

		m_shader->bind();
		m_shader->setUniform("depthCorrection", (int)bDepthCorrection);
		Shader::bindDefault();
	}

	//@ Toggle 3D filtering. Default is true.
	void	toggle3DFiltering()
	{
		b3DFiltering = !b3DFiltering;

		m_shader->bind();
		m_shader->setUniform("filter_3D", (int)b3DFiltering);
		Shader::bindDefault();
	}
};