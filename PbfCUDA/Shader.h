#pragma once

#include "gl\glew.h"
#include "glm\glm.hpp"
#include "shaderBasics.h"
#include <vector>
#include <map>

//@ OpenGL wrapper for shader unit.
class ShaderUnit : public GLObject{
protected:
	ShaderUnit(GLenum type, const char *code);
public:
	//@ Create shader unit with c-string.
	static ShaderUnit*	createWithString(GLenum type, const char *code);

	//@ Create shader unit with file.
	static ShaderUnit*	createWithFile(GLenum type, const char *filename);

	virtual				~ShaderUnit()		{ glDeleteShader(m_id); }

	//@ Get shader type(c-string) from GLenum(GL_VERTEX_SHADER / GL_FRAGMENT_SHADER / ...).
	static const char*	getShaderTypeString(GLenum type);
};

//@ OpenGL wrapper for shader program.
class Shader : public GLObject{
protected:
	std::map<std::string, GLint> m_uniformIDs;

	Shader(const std::vector<ShaderUnit *> &shaderUnits);
public:
	//@ Create shader program with shader units.
	static Shader* create(ShaderUnit *shaderUnit1,
						ShaderUnit *shaderUnit2 = nullptr,
						ShaderUnit *shaderUnit3 = nullptr);

	//@ Create shader program with files.
	static Shader* createWithFile(const char *vertex_shader_filename,
								const char *fragment_shader_filename,
								const char *geometry_shader_filename = nullptr);

	virtual			~Shader()			{ glDeleteProgram(m_id); }

	void			bind() const		{ glUseProgram(m_id); }
	static void		bindDefault()		{ glUseProgram(0); }

	//@ Get uniform ID using glUniformLocation().
	GLint			getUniformLocation(const char *location);

	// Uniform setters.
	//
	void			setUniform(const char *location, GLint value);
	void			setUniform(const char *location, GLuint value);
	void			setUniform(const char *location, GLfloat value);
	void			setUniform(const char *location, GLdouble value);
	void			setUniform(const char *location, const glm::vec2 &value);
	void			setUniform(const char *location, const glm::ivec2 &value);
	void			setUniform(const char *location, const glm::vec3 &value);
	void			setUniform(const char *location, const glm::vec4 &value);
	void			setUniform(const char *location, const glm::mat3 &value, GLboolean transpose = GL_FALSE);
	void			setUniform(const char *location, const glm::mat4 &value, GLboolean transpose = GL_FALSE);
};