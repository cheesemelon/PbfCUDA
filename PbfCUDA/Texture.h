#pragma once

#include "gl\glew.h"
#include "shaderBasics.h"
#include <vector>

struct ImageData{
	unsigned int width, height;
	GLint internalFormat;
	GLenum format;
	GLubyte *data;

	//@ Flip image vertically(upside-down).
	ImageData* flipVertical();

	//@ Create image data. (support BMP / JPG / GIF / PNG / TGA)
	static ImageData* create(const char *filename);

	//@ Create image data from raw image.
	static ImageData* createWithRawFile(const char *filename, int width, int height, int bytePerPixel);

	~ImageData() { delete[] data; }

protected:
	//@ Load image file with FreeImage. (BMP / JPG / GIF / PNG / TGA)
	ImageData(const char *filename);

	//@ Load raw image file.
	ImageData(const char *filename, int width, int height, int bytePerPixel);
};

//@ OpenGL wrapper for texture. Base class for various dimensions.
class Texture : public GLObject{
protected:
	GLenum m_target;

	Texture(GLenum target);
public:
	virtual ~Texture();

	void	bind() const;
	void	setWrapS(GLfloat mode) const;
	void	setWrapT(GLfloat mode) const;
	void	setWrap(GLfloat s, GLfloat t) const;
	void	setMinFilter(GLfloat mode) const;
	void	setMagFilter(GLfloat mode) const;
	void	setFilter(GLfloat min, GLfloat mag) const;
	void	generateMipmap() const;
};

//@ OpenGL wrapper for 1D texture.
class Texture1D : public Texture{
protected:
	unsigned int m_width;

	Texture1D(GLint internalFormat, unsigned int width, GLenum format, GLenum type, GLvoid *data);
public:
	//@ Create empty 1D texture.
	static Texture1D*	createEmpty(GLint internalFormat, unsigned int width, GLenum format, GLenum type);

	//@ Create 1D texture.
	static Texture1D*	create(ImageData *imageData);

	//@ Create 1D texture.
	static Texture1D*	createWithFile(const char *filename);

	virtual				~Texture1D()		{ }

	unsigned int		getWidth() const	{ return m_width; }
	static void			bindDefault()		{ glBindTexture(GL_TEXTURE_1D, 0); }
};

//@ OpenGL wrapper for 2D texture.
class Texture2D : public Texture{
protected:
	unsigned int m_width, m_height;

	Texture2D(GLint internalFormat, unsigned int width, unsigned int height, GLenum format, GLenum type, GLvoid *data);
public:
	//@ Create empty 2D texture.
	static Texture2D*	createEmpty(GLint internalFormat, unsigned int width, unsigned int height, GLenum format, GLenum type);

	//@ Create 2D texture.
	static Texture2D*	create(ImageData *imageData);

	//@ Create 2D texture.
	static Texture2D*	createWithFile(const char *filename);

	virtual				~Texture2D()		{ }

	unsigned int		getWidth() const	{ return m_width; }
	unsigned int		getHeight() const	{ return m_height; }
	static void			bindDefault()		{ glBindTexture(GL_TEXTURE_2D, 0); }
};

//@ OpenGL wrapper for cubemap texture.
class TextureCubemap : public Texture{
protected:
	unsigned int m_width, m_height;

	TextureCubemap(const std::vector<ImageData *> &imageDataSet);
public:
	//@ Create cubemap texture.
	static TextureCubemap*	create(ImageData *right, ImageData *left,
									ImageData *top, ImageData *bottom,
									ImageData *front, ImageData *back);

	//@ Create cubemap texture.
	static TextureCubemap*	createWithFiles(const char *right, const char *left,
											const char *top, const char *bottom,
											const char *front, const char *back);

	virtual					~TextureCubemap() { }

	unsigned int			getWidth() const		{ return m_width; }
	unsigned int			getHeight() const		{ return m_height; }
	static void				bindDefault()			{ glBindTexture(GL_TEXTURE_CUBE_MAP, 0); }
};