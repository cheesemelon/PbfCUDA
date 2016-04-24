
#include "Font.h"
#include "shaderBasics.h"
#include <string>
#include <vector>
#include <cstdarg>
#include <map>

Font::Font(GLuint quadBuffer, FontStyle style)
	: m_quadBuffer(quadBuffer)
{
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

	m_shader = Shader::create(ShaderUnit::createWithString(GL_VERTEX_SHADER, vertexShaderCode),
							ShaderUnit::createWithString(GL_FRAGMENT_SHADER, fragmentShaderCode));
	m_texture = Texture2D::createWithFile("resources/default16.png");
	m_texture->setWrap(GL_REPEAT, GL_REPEAT);
	m_texture->setFilter(GL_LINEAR, GL_LINEAR);
	checkOpenGL("Binding font texture", __FILE__, __LINE__, false, true);

	m_baseCharSize = glm::ivec2(m_texture->getWidth() / 16, m_texture->getHeight() / 16);
	m_color = glm::vec4(1.0f);
	setStyle(style);
}

void
Font::drawChar(int x, int y, char c)
{
	glm::ivec2 windowSize(1024, 768);
	drawChar(static_cast<float>(x) / static_cast<float>(windowSize.x), static_cast<float>(y) / static_cast<float>(windowSize.y), c);
}

void
Font::drawChar(float x, float y, char c)
{
	glm::vec2 windowsSize(1024.0f, 768.0f);
	glm::vec2 charSize = glm::vec2(m_baseCharSize) / windowsSize;

	m_shader->bind();
	{
		m_shader->setUniform("Color", m_color);
		m_shader->setUniform("StyleMask", m_styleMask);
		m_shader->setUniform("Position", glm::vec2(x, y));
		m_shader->setUniform("Char", static_cast<int>(c));
		m_shader->setUniform("BaseCharSize", m_baseCharSize);
		m_shader->setUniform("CharSize", charSize);

		glActiveTexture(GL_TEXTURE0);
		m_texture->bind();

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, m_quadBuffer);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);

		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

		glDisable(GL_BLEND);

		glDisableVertexAttribArray(0);
	}
	Shader::bindDefault();
}

void
Font::drawText(float x, float y, const char *text, ...)
{
	va_list args;
	va_start(args, text);
	char buff[1024];
	vsprintf_s(buff, text, args);
	va_end(args);

	glm::ivec2 windowSize(1024, 768);
	drawText(static_cast<int>(x * static_cast<float>(windowSize.x)), static_cast<int>(y * static_cast<float>(windowSize.y)), buff);
}

void
Font::drawText(int x, int y, const char *text, ...)
{
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

void
Font::setStyle(FontStyle style)
{
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