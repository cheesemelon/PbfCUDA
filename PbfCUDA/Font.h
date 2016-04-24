#pragma once

#include "gl\glew.h"
#include "glm\glm.hpp"
#include "Shader.h"
#include "Texture.h"

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
	Shader *m_shader;
	Texture2D *m_texture;
	const GLuint m_quadBuffer;

	Font(GLuint quadBuffer, FontStyle style);
public:
	static Font* create(GLuint quadBuffer, FontStyle style = FONT_STYLE_REGULAR)
	{
		return new Font(quadBuffer, style);
	}

	~Font()		{ delete m_shader, m_texture; }

	void drawChar(int x, int y, char c);
	void drawChar(float x, float y, char c);
	void drawText(int x, int y, const char *text, ...);
	void drawText(float x, float y, const char *text, ...);

	const glm::vec4&	getColor() const								{ return m_color; }
	void				setColor(int r, int g, int b, int a)			{ m_color = glm::vec4(r, g, b, a) * PER255F; }
	void				setColor(float r, float g, float b, float a)	{ m_color = glm::vec4(r, g, b, a); }
	FontStyle			getStyle() const								{ return m_style; }
	void				setStyle(FontStyle style);
};