#pragma once

#include "gl\glew.h"
#include "glm\glm.hpp"
#include "temp.h"
#include "shaderBasics.h"
#include <string>
#include <vector>
#include <cstdarg>
#include <map>

enum FontStyle {
	FONT_STYLE_REGULAR, FONT_STYLE_BOLD,
	FONT_STYLE_ITALIC, FONT_STYLE_BOLD_ITALIC,
};

class Font{
private:
	glm::ivec2 m_baseCharSize;
	glm::vec4 m_color;
	FontStyle m_style;
	glm::vec4 m_styleMask;
	GLuint m_programID;
	GLuint m_textureID;
	GLuint m_vertexBuffer, m_uvBuffer;
	std::map<std::string, GLuint> m_uniforms;
public:
	Font(FontStyle style = FONT_STYLE_REGULAR) {
		const char *vertexShaderCode =
			"#version 330 core\n \
			layout(location = 0) in vec2 vertexPosition_screenspace; \
			noperspective out vec2 UV; \
			uniform sampler2D FontTexture; \
			uniform vec2 Position; \
			uniform int Char; \
			uniform ivec2 BaseCharSize; \
			uniform vec2 CharSize; \
			void main(){ \
				vec2 letterScale = vec2(BaseCharSize) / vec2(textureSize(FontTexture, 0)); \
				vec2 offset = vec2(float(Char % BaseCharSize.x) / float(BaseCharSize.x), \
				float(Char / BaseCharSize.y) / float(BaseCharSize.y)); \
				vec2 coord = (vertexPosition_screenspace * 0.5 + 0.5); \
				UV = coord * letterScale + offset; \
				offset = Position * 2.0 - 1.0 + CharSize; \
				offset.y = -offset.y; \
				gl_Position = vec4(vertexPosition_screenspace * CharSize + offset, 0.0, 1.0); \
			}";
		const char *fragmentShaderCode =
			"#version 330 core\n \
			noperspective in vec2 UV; \
			out vec4 FragColor; \
			uniform sampler2D FontTexture; \
			uniform vec4 Color = vec4(1.0); \
			uniform vec4 StyleMask = vec4(0.0, 1.0, 0.0, 0.0); \
			void main(){ \
				vec4 letter = texture(FontTexture, UV) * StyleMask; \
				FragColor = Color * vec4(max(max(max(letter.r, letter.g), letter.b), letter.a)); \
			}";

		GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShaderID, 1, &vertexShaderCode, NULL);
		glCompileShader(vertexShaderID);

		GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShaderID, 1, &fragmentShaderCode, NULL);
		glCompileShader(fragmentShaderID);

		m_programID = glCreateProgram();
		glAttachShader(m_programID, vertexShaderID);
		glAttachShader(m_programID, fragmentShaderID);
		glLinkProgram(m_programID);

		glDeleteShader(vertexShaderID);
		glDeleteShader(fragmentShaderID);

		GLint result = GL_FALSE;
		int infoLogLength = 0;
		glGetProgramiv(m_programID, GL_LINK_STATUS, &result);
		glGetProgramiv(m_programID, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (result == GL_FALSE){
			char *errorMessage = new char[infoLogLength];
			glGetProgramInfoLog(m_programID, infoLogLength, NULL, errorMessage);
			std::cout << errorMessage << std::endl;
			delete[] errorMessage;
		}

		m_uniforms["FontTexture"] = glGetUniformLocation(m_programID, "FontTexture");
		m_uniforms["Color"] = glGetUniformLocation(m_programID, "Color");
		m_uniforms["StyleMask"] = glGetUniformLocation(m_programID, "StyleMask");
		m_uniforms["Position"] = glGetUniformLocation(m_programID, "Position");
		m_uniforms["Char"] = glGetUniformLocation(m_programID, "Char");
		m_uniforms["BaseCharSize"] = glGetUniformLocation(m_programID, "BaseCharSize");
		m_uniforms["CharSize"] = glGetUniformLocation(m_programID, "CharSize");

		std::vector<GLubyte> data;
		unsigned int width, height;
		loadImage("default16.png", &data, &width, &height);
		glGenTextures(1, &m_textureID);
		glBindTexture(GL_TEXTURE_2D, m_textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, (GLvoid *)&(data.front()));
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		checkOpenGL("Binding font texture", __FILE__, __LINE__, false, true);

		m_baseCharSize = glm::ivec2(width / 16, height / 16);
		m_color = glm::vec4(1.0f);
		setStyle(style);

		glGenBuffers(1, &m_vertexBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GlobalVariables::quad_vertexData), GlobalVariables::quad_vertexData, GL_STATIC_DRAW);

		glGenBuffers(1, &m_uvBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, m_uvBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GlobalVariables::quad_uvData), GlobalVariables::quad_uvData, GL_STATIC_DRAW);
	}

	~Font(){
		glDeleteBuffers(1, &m_vertexBuffer);
		glDeleteBuffers(1, &m_uvBuffer);
		glDeleteProgram(m_programID);
		glDeleteTextures(1, &m_textureID);
	}

	inline void drawChar(int x, int y, char c){
		glm::ivec2 windowSize(1024, 768);
		drawChar(static_cast<float>(x) / static_cast<float>(windowSize.x), static_cast<float>(y) / static_cast<float>(windowSize.y), c);
	}

	void drawChar(float x, float y, char c){
		glm::vec2 windowsSize(1024.0f, 768.0f);
		glm::vec2 charSize = glm::vec2(m_baseCharSize) / windowsSize;

		glUseProgram(m_programID);
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, m_textureID);
			glUniform1i(m_uniforms["FontTexture"], 0);
			glUniform4f(m_uniforms["Color"], m_color.r, m_color.g, m_color.b, m_color.a);
			glUniform4f(m_uniforms["StyleMask"], m_styleMask.x, m_styleMask.y, m_styleMask.z, m_styleMask.w);
			glUniform2f(m_uniforms["Position"], x, y);
			glUniform1i(m_uniforms["Char"], static_cast<int>(c));
			glUniform2i(m_uniforms["BaseCharSize"], m_baseCharSize.x, m_baseCharSize.y);
			glUniform2f(m_uniforms["CharSize"], charSize.x, charSize.y);

			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);

			glEnable(GL_BLEND);
			glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

			glDisable(GL_BLEND);

			glDisableVertexAttribArray(0);
		}
	}

	void drawText(float x, float y, const char *text, ...){
		va_list args;
		va_start(args, text);
		char buff[1024];
		vsprintf_s(buff, text, args);
		va_end(args);

		glm::ivec2 windowSize(1024, 768);
		drawText(static_cast<int>(x * static_cast<float>(windowSize.x)), static_cast<int>(y * static_cast<float>(windowSize.y)), buff);
	}

	void drawText(int x, int y, const char *text, ...){
		va_list args;
		va_start(args, text);
		char buff[1024];
		vsprintf_s(buff, text, args);
		va_end(args);

		int _x = x;
		char *c = buff;
		while (*c != '\0'){
			if (*c == '\n'){
				_x = x;
				y += m_baseCharSize.y;
			}
			else{
				drawChar(_x, y, *c);
				_x += m_baseCharSize.x * 0.7;
			}

			++c;
		}
	}

	const glm::vec4& getColor() const { return m_color; }
	void setColor(int r, int g, int b, int a){
		m_color = glm::vec4(r, g, b, a) * PER255F;
	}
	void setColor(float r, float g, float b, float a){
		m_color = glm::vec4(r, g, b, a);
	}

	FontStyle getStyle() const { return m_style; }
	void setStyle(FontStyle style){
		m_style = style;
		switch (style)
		{
		case FONT_STYLE_REGULAR:
			m_styleMask = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
			break;
		case FONT_STYLE_BOLD:
			m_styleMask = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
			break;
		case FONT_STYLE_ITALIC:
			m_styleMask = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
			break;
		case FONT_STYLE_BOLD_ITALIC:
			m_styleMask = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
			break;
		default:
			break;
		}
	}
};