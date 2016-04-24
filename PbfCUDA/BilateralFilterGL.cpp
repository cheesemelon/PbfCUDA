#include "BilateralFilterGL.h"

#include <vector>
#include "gl\glew.h"
#include "glm\glm.hpp"
#include "Shader.h"
#include "Texture.h"

BilateralFilterGL::BilateralFilterGL(int width, int height,
									int radius, float sigma_s, float sigma_r, int nIterations,
									const glm::mat4 &P, const glm::mat4 &invP)
	: m_width(width), m_height(height),
	m_radius(radius), m_sigma_s(sigma_s), m_sigma_r(sigma_r), m_nIterations(nIterations),
	bDepthCorrection(true), b3DFiltering(true)
{
	// Create compute shader(bilateral filter).
	m_shader = Shader::create(ShaderUnit::createWithFile(GL_COMPUTE_SHADER, "BilateralFilter.cs.glsl"));
	m_shader->setUniform("width", m_width);
	m_shader->setUniform("height", m_height);
	m_shader->setUniform("sigma_s", m_sigma_s);
	m_shader->setUniform("sigma_r", m_sigma_r);
	m_shader->setUniform("P32", P[3][2]);
	m_shader->setUniform("P22", P[2][2]);
	m_shader->setUniform("invP", invP);

	m_shader->setUniform("depthCorrection", 1);
	m_shader->setUniform("filter_3D", 1);
	Shader::bindDefault();

	// Create temporal texture.
	m_tex_canvas = Texture2D::createEmpty(GL_R32F, width, height, GL_RED, GL_FLOAT);
	m_tex_canvas->setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
	m_tex_canvas->setFilter(GL_LINEAR, GL_LINEAR);
	Texture2D::bindDefault();
}

void
BilateralFilterGL::setParameters(int radius, float sigma_s, float sigma_r, int nIterations)
{
	m_radius = radius;
	m_sigma_s = sigma_s;
	m_sigma_r = sigma_r;
	m_nIterations = nIterations;

	m_shader->bind();
	m_shader->setUniform("sigma_s", sigma_s);
	m_shader->setUniform("sigma_r", sigma_r);
	Shader::bindDefault();

	//printf("BFGL r : %d, sigma_s : %.2f, sigma_r : %.2f\n", m_radius, m_sigma_s, m_sigma_r);
}

void
BilateralFilterGL::run(GLuint depthTextureID)
{
	m_shader->bind();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, depthTextureID);

	glActiveTexture(GL_TEXTURE1);
	glBindImageTexture(1, m_tex_canvas->getID(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

	int radius = m_radius;
	for (int i = 0; i < m_nIterations; ++i){
		m_shader->setUniform("radius", radius);

		// Run filter vertically.
		m_shader->setUniform("vertical", 1);
		glDispatchCompute((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16, 1);
		glCopyImageSubData(m_tex_canvas->getID(), GL_TEXTURE_2D, 0, 0, 0, 0,
							depthTextureID, GL_TEXTURE_2D, 0, 0, 0, 0,
							m_width, m_height, 1);

		// Run filter horizontally.
		m_shader->setUniform("vertical", 0);
		glDispatchCompute((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16, 1);
		glCopyImageSubData(m_tex_canvas->getID(), GL_TEXTURE_2D, 0, 0, 0, 0,
							depthTextureID, GL_TEXTURE_2D, 0, 0, 0, 0,
							m_width, m_height, 1);

		// Halve kernel radius.
		radius = (radius >> 1);
	}

	Shader::bindDefault();
	checkOpenGL("Error on the filter using compute shader.", __FILE__, __LINE__, true, true);
}
