#include "Shader.h"
#include <iostream>
#include <fstream>
#include <string>

/////////////////////////////////////////////////////////////////////////////////
// ShaderUnit implementations

ShaderUnit::ShaderUnit(GLenum type, const char *code)
{
	// Create shader from code.
	m_id = glCreateShader(type);
	glShaderSource(m_id, 1, &code, nullptr);
	glCompileShader(m_id);
}

ShaderUnit*
ShaderUnit::createWithString(GLenum type, const char *code)
{
	std::cout << "Compiling " << getShaderTypeString(type) << " shader with string..." << std::endl;
	return new ShaderUnit(type, code);
}

ShaderUnit*
ShaderUnit::createWithFile(GLenum type, const char *filename)
{
	// Read the shader code from the file.
	std::vector<char> code;
	std::ifstream stream(filename, std::ios::binary | std::ios::ate);
	if (stream.is_open()){
		unsigned int fileSize = stream.tellg();
		stream.close();
		stream.clear();
		stream.open(filename, std::ios::binary);
		code.resize(fileSize + 1);
		stream.read(code.data(), fileSize);
		stream.close();
	}
	else{
		std::cerr << "Impossible to open \"" << filename << "\". Are you in the right directory?" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::cout << "Compiling " << getShaderTypeString(type) << " shader with \"" << filename << "\"..." << std::endl;
	return new ShaderUnit(type, code.data());
}

const char *
ShaderUnit::getShaderTypeString(GLenum type)
{
	switch (type)
	{
		case GL_VERTEX_SHADER:			return "vertex";
		case GL_FRAGMENT_SHADER:		return "fragment";
		case GL_GEOMETRY_SHADER:		return "geometry";
		case GL_COMPUTE_SHADER:			return "compute";
		default:
			std::cerr << "Compiling shader failure. Unsupportable shader type." << std::endl;
			exit(EXIT_FAILURE);
	}
}

/////////////////////////////////////////////////////////////////////////////////
// Shader implementations

Shader::Shader(const std::vector<ShaderUnit *> &shaderUnits)
{
	// Link the program.
	std::cout << "Linking program... ";
	m_id = glCreateProgram();
	for (auto &shaderUnit : shaderUnits){
		glAttachShader(m_id, shaderUnit->getID());
	}
	glLinkProgram(m_id);

	// Check the program
	GLint result = GL_FALSE;
	int infoLogLength;
	glGetProgramiv(m_id, GL_LINK_STATUS, &result);
	glGetProgramiv(m_id, GL_INFO_LOG_LENGTH, &infoLogLength);
	if (result == GL_FALSE){
		std::vector<char> errorMessage(infoLogLength);
		glGetProgramInfoLog(m_id, infoLogLength, nullptr, &errorMessage[0]);
		std::cout << std::endl;
		std::cerr << &errorMessage[0] << std::endl;
		std::cerr << "Compling shader failed." << std::endl;
		exit(EXIT_FAILURE);
	}
	glUseProgram(m_id);
	std::cout << "Done." << std::endl;
}

Shader*
Shader::create(ShaderUnit *shaderUnit1, ShaderUnit *shaderUnit2, ShaderUnit *shaderUnit3)
{
	std::vector<ShaderUnit *> shaderUnits;
	shaderUnits.push_back(shaderUnit1);
	if (shaderUnit2 != nullptr){ shaderUnits.push_back(shaderUnit2); }
	if (shaderUnit3 != nullptr){ shaderUnits.push_back(shaderUnit3); }
	Shader *instance = new Shader(shaderUnits);
	for (auto &shaderUnit : shaderUnits){
		delete shaderUnit;
	}
	return instance;
}

Shader*
Shader::createWithFile(const char *vertex_shader_filename,
							const char *fragment_shader_filename,
							const char *geometry_shader_filename)
{
	std::vector<ShaderUnit *> shaderUnits;
	shaderUnits.push_back(ShaderUnit::createWithFile(GL_VERTEX_SHADER, vertex_shader_filename));
	shaderUnits.push_back(ShaderUnit::createWithFile(GL_FRAGMENT_SHADER, fragment_shader_filename));
	if (geometry_shader_filename != nullptr){
		shaderUnits.push_back(ShaderUnit::createWithFile(GL_GEOMETRY_SHADER, geometry_shader_filename));
	}
	Shader *instance = new Shader(shaderUnits);
	for (auto &shaderUnit : shaderUnits){
		delete shaderUnit;
	}
	return instance;
}

GLint
Shader::getUniformLocation(const char *location)
{
	auto &found = m_uniformIDs.find(location);
	if (found == m_uniformIDs.end()){
		GLint uniformID = glGetUniformLocation(m_id, location);
		m_uniformIDs[location] = uniformID;
		//std::cout << "New uniform location \"" << location << "\" is set." << std::endl;
		return uniformID;
	}
	return found->second;
}

void
Shader::setUniform(const char *location, GLint value)
{
	glUniform1i(getUniformLocation(location), value);
}

void
Shader::setUniform(const char *location, GLuint value)
{
	glUniform1ui(getUniformLocation(location), value);
}

void
Shader::setUniform(const char *location, GLfloat value)
{
	glUniform1f(getUniformLocation(location), value);
}

void
Shader::setUniform(const char *location, GLdouble value)
{
	glUniform1d(getUniformLocation(location), value);
}

void
Shader::setUniform(const char *location, const glm::vec2 &value)
{
	glUniform2f(getUniformLocation(location), value.x, value.y);
}

void
Shader::setUniform(const char *location, const glm::ivec2 &value)
{
	glUniform2i(getUniformLocation(location), value.x, value.y);
}

void
Shader::setUniform(const char *location, const glm::vec3 &value)
{
	glUniform3f(getUniformLocation(location), value.x, value.y, value.z);
}

void
Shader::setUniform(const char *location, const glm::vec4 &value)
{
	glUniform4f(getUniformLocation(location), value.x, value.y, value.z, value.w);
}

void
Shader::setUniform(const char *location, const glm::mat3 &value, GLboolean transpose)
{
	glUniformMatrix3fv(getUniformLocation(location), 1, transpose, &(value[0][0]));
}

void
Shader::setUniform(const char *location, const glm::mat4 &value, GLboolean transpose)
{
	glUniformMatrix4fv(getUniformLocation(location), 1, transpose, &(value[0][0]));
}