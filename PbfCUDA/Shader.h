#pragma once

#include "gl\glew.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class ShaderUnit {
protected:
	GLuint m_id;

	ShaderUnit(GLenum type, const char *code){
		// Create shader from code.
		m_id = glCreateShader(type);
		glShaderSource(m_id, 1, &code, NULL);
		glCompileShader(m_id);
	}
public:
	static ShaderUnit* createWithString(GLenum type, const char *code){
		std::string strType;
		switch (type)
		{
		case GL_VERTEX_SHADER:
			strType = "vertex";
			break;
		case GL_FRAGMENT_SHADER:
			strType = "fragment";
			break;
		case GL_GEOMETRY_SHADER:
			strType = "geometry";
			break;
		case GL_COMPUTE_SHADER:
			strType = "compute";
			break;
		default:
			std::cout << "Compiling shader failure. Unsupported shader type." << std::endl;
			return NULL;
		}
		std::cout << "Compiling " << strType << " shader with string..." << std::endl;

		return new ShaderUnit(type, code);
	}

	static ShaderUnit* createWithFile(GLenum type, const char *filename){
		// Read the shader code from the file.
		std::string code;
		std::ifstream stream(filename, std::ios::in);
		if (stream.is_open()){
			std::string line = "";
			while (std::getline(stream, line)){
				code += "\n" + line;
			}
			stream.close();
		}
		else{
			std::cerr << "Impossible to open \"" << filename << "\". Are you in the right directory?" << std::endl;
			exit(EXIT_FAILURE);
		}

		std::string strType;
		switch (type)
		{
		case GL_VERTEX_SHADER:
			strType = "vertex";
			break;
		case GL_FRAGMENT_SHADER:
			strType = "fragment";
			break;
		case GL_GEOMETRY_SHADER:
			strType = "geometry";
			break;
		case GL_COMPUTE_SHADER:
			strType = "compute";
			break;
		default:
			std::cerr << "Compiling shader failure. Unsupported shader type." << std::endl;
			exit(EXIT_FAILURE);
		}

		// Compile vertex shader.
		std::cout << "Compiling " << strType << " shader with \"" << filename << "\"..." << std::endl;

		return new ShaderUnit(type, code.c_str());
	}

	~ShaderUnit(){
		glDeleteShader(m_id);
	}

	GLuint getID() const {
		return m_id;
	}
};

class Shader {
protected:
	GLuint m_id;

	Shader(const std::vector<ShaderUnit *> &shaderUnits){
		// Link the program.
		std::cout << "Linking program..." << std::endl;
		m_id = glCreateProgram();
		for(auto &shaderUnit : shaderUnits){
			glAttachShader(m_id, shaderUnit->getID());
		}
		glLinkProgram(m_id);

		// Check the program
		GLint result = GL_FALSE;
		int infoLogLength;
		glGetProgramiv(m_id, GL_LINK_STATUS, &result);
		glGetProgramiv(m_id, GL_INFO_LOG_LENGTH, &infoLogLength);
		//if (infoLogLength > 1){
		if (result == GL_FALSE){
			std::vector<char> errorMessage(infoLogLength);
			glGetProgramInfoLog(m_id, infoLogLength, nullptr, &errorMessage[0]);
			std::cout << &errorMessage[0] << std::endl;
			std::cerr << "Compiling shader program." << std::endl;
			exit(EXIT_FAILURE);
		}

		for (auto &shaderunit : shaderUnits){
			delete shaderunit;
		}
	}
public:
	static Shader* create(ShaderUnit *shaderUnit1, ShaderUnit *shaderUnit2 = NULL, ShaderUnit *shaderUnit3 = NULL){
		std::vector<ShaderUnit *> shaderUnits;
		shaderUnits.push_back(shaderUnit1);
		if (shaderUnit2 != NULL){
			shaderUnits.push_back(shaderUnit2);
		}
		if (shaderUnit3 != NULL){
			shaderUnits.push_back(shaderUnit3);
		}
		return new Shader(shaderUnits);
	}

	~Shader(){
		glDeleteProgram(m_id);
	}

	GLint getUniformLocation(const char *uniformLocation) const{
		return glGetUniformLocation(m_id, uniformLocation);
	}

	GLuint getID() const {
		return m_id;
	}
};