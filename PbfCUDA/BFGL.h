#pragma once

#include <vector>
#include "gl\glew.h"
#include "glm\glm.hpp"
#include "temp.h"
#include "Shader.h"

class LowpassFilter;

class BFGL{
protected:
	int m_width, m_height;
	int m_radius;
	float m_sigma_s, m_sigma_r;
	int m_nIterations;
	GLuint m_canvasTextureID;
	Shader *m_shader;

	LowpassFilter *m_lowpassFilter;

	BFGL(int width, int height,
		int radius, float sigma_s, float sigma_r, int nIterations)
		: m_width(width), m_height(height),
		m_radius(radius), m_sigma_s(sigma_s), m_sigma_r(sigma_r), m_nIterations(nIterations)
	{
		m_shader = Shader::create(ShaderUnit::createWithFile(GL_COMPUTE_SHADER, "BilateralFilter.frag"));

		glGenTextures(1, &m_canvasTextureID);
		glBindTexture(GL_TEXTURE_2D, m_canvasTextureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
public:
	static BFGL* create(int width, int height, int radius, float sigma_s, float sigma_r, int nIterations)
	{
		return new BFGL(width, height, radius, sigma_s, sigma_r, nIterations);
	}

	~BFGL() {
		delete m_shader;
		glDeleteTextures(1, &m_canvasTextureID);
	}

	void setParameters(int radius, float sigma_s, float sigma_r, int nIterations, float zNear, float zFar){
		m_radius = radius;
		m_sigma_s = sigma_s;
		m_sigma_r = sigma_r * (zFar - zNear) / 255.0f;
		m_nIterations = nIterations;

		printf("BFGL r : %d, sigma_s : %.2f, sigma_r : %.2f\n", m_radius, m_sigma_s, m_sigma_r);
	}

	void run(GLuint depthTextureID, const glm::mat4 &P, const glm::mat4 &invP){
		glUseProgram(m_shader->getID());

		glUniformMatrix4fv(m_shader->getUniformLocation("P"), 1, GL_FALSE, &P[0][0]);
		glUniformMatrix4fv(m_shader->getUniformLocation("invP"), 1, GL_FALSE, &invP[0][0]);
		glUniform1i(m_shader->getUniformLocation("width"), m_width);
		glUniform1i(m_shader->getUniformLocation("height"), m_height);
		glUniform1f(m_shader->getUniformLocation("sigma_s"), m_sigma_s);
		glUniform1f(m_shader->getUniformLocation("sigma_r"), m_sigma_r);

		glActiveTexture(GL_TEXTURE0);
		glBindImageTexture(0, depthTextureID, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

		glActiveTexture(GL_TEXTURE1);
		glBindImageTexture(1, m_canvasTextureID, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

		int radius = m_radius;
		int vertical = 0;
		for (int i = 0; i < m_nIterations; ++i){
			glUniform1i(m_shader->getUniformLocation("radius"), radius);
			glUniform1i(m_shader->getUniformLocation("vertical"), vertical);
			glDispatchCompute((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16, 1);
			checkOpenGL("Filter error.", __FILE__, __LINE__, true, true);

			glCopyImageSubData(m_canvasTextureID, GL_TEXTURE_2D, 0, 0, 0, 0,
				depthTextureID, GL_TEXTURE_2D, 0, 0, 0, 0,
				m_width, m_height, 1);

			vertical = 1 - vertical;
			radius = (radius >> 1);
		}

		glUseProgram(0);
	}
};