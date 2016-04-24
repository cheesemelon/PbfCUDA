#include "Texture.h"
#include "FreeImage\FreeImage.h"
#include <iostream>
#include <fstream>
#include <cassert>

/////////////////////////////////////////////////////////////////////////////////
// ImageData implementations

ImageData*
ImageData::flipVertical(){
	int bytePerPixel;
	switch (internalFormat){
		case GL_RED:	bytePerPixel = 1;	break;
		case GL_RGB:	bytePerPixel = 3;	break;
		case GL_RGBA:	bytePerPixel = 4;	break;
		default:
		{
			std::cerr << "Unsupportable format: flipVertical()" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	std::vector<GLubyte> horizontal(width * bytePerPixel);
	for (int y = 0; y < height / 2; ++y){
		GLubyte *frontLoc = data + y * bytePerPixel * width;
		GLubyte *backLoc = data + (height - 1 - y) * bytePerPixel * width;
		memcpy(horizontal.data(), frontLoc, width * bytePerPixel);
		memcpy(frontLoc, backLoc, width * bytePerPixel);
		memcpy(backLoc, horizontal.data(), width * bytePerPixel);
	}

	return this;
}

ImageData*
ImageData::create(const char *filename)
{
	return new ImageData(filename);
}

ImageData*
ImageData::createWithRawFile(const char *filename, int width, int height, int bytePerPixel)
{
	return new ImageData(filename, width, height, bytePerPixel);
}

ImageData::ImageData(const char *filename)
{
	assert(sizeof(BYTE) == sizeof(GLubyte));

	// image format
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	// pointer to the image, once loaded
	FIBITMAP *dib(0);
	// pointer to the image data
	BYTE* bits(0);

	// check the file signature and deduce its format
	fif = FreeImage_GetFileType(filename, 0);
	// if still unknown, try to guess the file format from the file extension
	if (fif == FIF_UNKNOWN){
		fif = FreeImage_GetFIFFromFilename(filename);
	}
	// if still unkown, return failure
	if (fif == FIF_UNKNOWN){
		std::cerr << "Loading " << filename << " failure. fif unknown." << std::endl;
		exit(EXIT_FAILURE);
	}
	// If image foramt is unsupportable, return failure.
	if (fif != FIF_BMP && fif != FIF_JPEG && fif != FIF_GIF && fif != FIF_PNG && fif != FIF_TARGA){
		std::cerr << "Loading " << filename << " failure. Unsupportable format." << std::endl;
		exit(EXIT_FAILURE);
	}

	// check that the plugin has reading capabilities and load the file
	if (FreeImage_FIFSupportsReading(fif))
		dib = FreeImage_Load(fif, filename);
	// if the image failed to load, return failure
	if (!dib){
		std::cerr << "Loading " << filename << " failure. dib error." << std::endl;
		exit(EXIT_FAILURE);
	}

	unsigned int bitPerPixel = FreeImage_GetBPP(dib);
	switch (bitPerPixel)
	{
		case 8:		internalFormat = GL_RED;	format = GL_RED;	break;
		case 24:	internalFormat = GL_RGB;	format = GL_BGR;	break;
		case 32:	internalFormat = GL_RGBA;	format = GL_BGRA;	break;
		default:
			FreeImage_Unload(dib);
			std::cerr << "Loading " << filename << " failure. Unsupportable format." << std::endl;
			exit(EXIT_FAILURE);
	}

	// retrieve the image data
	bits = FreeImage_GetBits(dib);
	// get the image width and height
	width = FreeImage_GetWidth(dib);
	height = FreeImage_GetHeight(dib);
	// if this somehow one of these failed (they shouldn't), return failure
	if ((bits == 0) || (width == 0) || (height == 0)){
		std::cerr << "Loading " << filename << " failure. error unknown." << std::endl;
		exit(EXIT_FAILURE);
	}

	unsigned int dataSize = FreeImage_GetDIBSize(dib);
	data = new GLubyte[dataSize];
	memcpy(data, bits, dataSize);

	FreeImage_Unload(dib);
}

ImageData::ImageData(const char *filename, int width, int height, int bytePerPixel)
{
	this->width = width;
	this->height = height;

	switch (bytePerPixel)
	{
		case 1:		this->internalFormat = this->format = GL_RED;		break;
		case 3:		this->internalFormat = this->format = GL_RGB;		break;
		case 4:		this->internalFormat = this->format = GL_RGBA;		break;
		default:
		{
			std::cerr << "Can't accept this image type: byte per pixel." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	std::ifstream stream(filename, std::ios::binary);
	if (!stream.is_open()){
		std::cerr << "Can't open " << filename << "." << std::endl;
		exit(EXIT_FAILURE);
	}
	data = new GLubyte[width * height * bytePerPixel];
	stream.read(reinterpret_cast<char *>(data), sizeof(GLubyte) * width * height * bytePerPixel);
}

/////////////////////////////////////////////////////////////////////////////////
// Texture implementations

Texture::Texture(GLenum target)
	: m_target(target)
{
	glGenTextures(1, &m_id);
	glBindTexture(m_target, m_id);
}

Texture::~Texture()
{
	glDeleteTextures(1, &m_id);
}

void
Texture::bind() const
{
	glBindTexture(m_target, m_id);
}

void
Texture::setWrapS(GLfloat mode) const
{
	glTexParameterf(m_target, GL_TEXTURE_WRAP_S, mode);
}

void
Texture::setWrapT(GLfloat mode) const
{
	glTexParameterf(m_target, GL_TEXTURE_WRAP_T, mode);
}

void
Texture::setWrap(GLfloat s, GLfloat t) const
{
	glTexParameterf(m_target, GL_TEXTURE_WRAP_S, s);
	glTexParameterf(m_target, GL_TEXTURE_WRAP_T, t);
}

void
Texture::setMinFilter(GLfloat mode) const
{
	glTexParameterf(m_target, GL_TEXTURE_MIN_FILTER, mode);
}

void
Texture::setMagFilter(GLfloat mode) const
{
	glTexParameterf(m_target, GL_TEXTURE_MAG_FILTER, mode);
}

void
Texture::setFilter(GLfloat min, GLfloat mag) const
{
	glTexParameterf(m_target, GL_TEXTURE_MIN_FILTER, min);
	glTexParameterf(m_target, GL_TEXTURE_MAG_FILTER, mag);
}

void
Texture::generateMipmap() const
{
	glGenerateMipmap(m_target);
}

/////////////////////////////////////////////////////////////////////////////////
// Texture1D implementations

Texture1D::Texture1D(GLint internalFormat, unsigned int width, GLenum format, GLenum type, GLvoid *data)
	: Texture(GL_TEXTURE_1D), m_width(width)
{
	glTexImage1D(GL_TEXTURE_1D, 0, internalFormat, m_width, 0, format, type, data);
	glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

Texture1D*
Texture1D::createEmpty(GLint internalFormat, unsigned int width, GLenum format, GLenum type)
{
	return new Texture1D(internalFormat, width, format, type, nullptr);
}

Texture1D*
Texture1D::create(ImageData *imageData)
{
	Texture1D *instance = new Texture1D(imageData->internalFormat, imageData->width, imageData->format, GL_UNSIGNED_BYTE, (GLvoid *)imageData->data);
	delete imageData;
	return instance;
}

Texture1D*
Texture1D::createWithFile(const char *filename)
{
	ImageData *imageData = ImageData::create(filename);
	Texture1D *instance = new Texture1D(imageData->internalFormat, imageData->width, imageData->format, GL_UNSIGNED_BYTE, (GLvoid *)imageData->data);
	delete imageData;
	return instance;
}

/////////////////////////////////////////////////////////////////////////////////
// Texture2D implementations

Texture2D::Texture2D(GLint internalFormat, unsigned int width, unsigned int height, GLenum format, GLenum type, GLvoid *data)
	: Texture(GL_TEXTURE_2D), m_width(width), m_height(height) {
	// This is for an not aligned image data. No need with FreeImage.
	//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	//glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
	//glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
	//glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, m_width, m_height, 0, format, type, data);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

Texture2D*
Texture2D::createEmpty(GLint internalFormat, unsigned int width, unsigned int height, GLenum format, GLenum type)
{
	return new Texture2D(internalFormat, width, height, format, type, nullptr);
}

Texture2D*
Texture2D::create(ImageData *imageData)
{
	Texture2D *instance = new Texture2D(imageData->internalFormat, imageData->width, imageData->height, imageData->format, GL_UNSIGNED_BYTE, imageData->data);
	delete imageData;
	return instance;
}

Texture2D*
Texture2D::createWithFile(const char *filename)
{
	ImageData *imageData = ImageData::create(filename);
	Texture2D *instance = new Texture2D(imageData->internalFormat, imageData->width, imageData->height, imageData->format, GL_UNSIGNED_BYTE, imageData->data);
	delete imageData;
	return instance;
}

/////////////////////////////////////////////////////////////////////////////////
// TextureCubemap implementations

TextureCubemap::TextureCubemap(const std::vector<ImageData *> &imageDataSet)
	: Texture(GL_TEXTURE_CUBE_MAP), m_width(imageDataSet[0]->width), m_height(imageDataSet[0]->height)
{
	for (int i = 0; i < 6; ++i){
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, imageDataSet[i]->internalFormat, imageDataSet[i]->width, imageDataSet[i]->height, 0, imageDataSet[i]->format, GL_UNSIGNED_BYTE, imageDataSet[i]->data);
	}
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

TextureCubemap*
TextureCubemap::create(ImageData *right, ImageData *left,
						ImageData *top, ImageData *bottom,
						ImageData *front, ImageData *back)
{
	std::vector<ImageData *> imageDataSet;
	imageDataSet.reserve(6);
	// Cubemap texture is somehow upside-down, so the image needs to be flipped vertically.
	imageDataSet.push_back(right->flipVertical());
	imageDataSet.push_back(left->flipVertical());
	imageDataSet.push_back(top->flipVertical());
	imageDataSet.push_back(bottom->flipVertical());
	imageDataSet.push_back(front->flipVertical());
	imageDataSet.push_back(back->flipVertical());
	TextureCubemap *instance = new TextureCubemap(imageDataSet);
	for (auto &imageData : imageDataSet){
		delete imageData;
	}
	return instance;
}

TextureCubemap*
TextureCubemap::createWithFiles(const char *right, const char *left,
								const char *top, const char *bottom,
								const char *front, const char *back)
{
	std::vector<ImageData *> imageDataSet;
	imageDataSet.reserve(6);
	// Cubemap texture is somehow upside-down, so the image needs to be flipped vertically.
	imageDataSet.push_back(ImageData::create(right)->flipVertical());
	imageDataSet.push_back(ImageData::create(left)->flipVertical());
	imageDataSet.push_back(ImageData::create(top)->flipVertical());
	imageDataSet.push_back(ImageData::create(bottom)->flipVertical());
	imageDataSet.push_back(ImageData::create(front)->flipVertical());
	imageDataSet.push_back(ImageData::create(back)->flipVertical());
	TextureCubemap *instance = new TextureCubemap(imageDataSet);
	for (auto &imageData : imageDataSet){
		delete imageData;
	}
	return instance;
}