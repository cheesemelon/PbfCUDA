#pragma once

#include "gl\glew.h"
#include "FreeImage\FreeImage.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <limits>

#pragma comment (lib, "FreeImage\\FreeImage.lib")

#define PER255F 0.003921568627f	//@ /255.0f

namespace GlobalVariables {
	const GLfloat quad_vertexData[8] = {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f, 1.0f,
		-1.0f, 1.0f
	};

	const GLfloat quad_uvData[8] = {
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f
	};
}

extern const char *_cudaGetErrorEnum(cudaError_t error);

static void saveAsPNG(const char *filename, const GLubyte *data, int width, int height){
	BYTE *pixels = (BYTE *)data;

	FIBITMAP *image = FreeImage_ConvertFromRawBits(pixels, width, height, width, 8, 0x0000FF, 0xFF0000, 0x00FF00, false);
	FreeImage_Save(FIF_PNG, image, filename, 0);

	FreeImage_Unload(image);
}

static void saveAsPNG(const char *filename, const GLfloat *data, int width, int height){
	BYTE *pixels = new BYTE[width * height];
	for (int i = 0; i < width * height; ++i){
		pixels[i] = 255 * data[i];
	}

	FIBITMAP *image = FreeImage_ConvertFromRawBits(pixels, width, height, width, 8, 0x0000FF, 0xFF0000, 0x00FF00, false);
	FreeImage_Save(FIF_PNG, image, filename, 0);

	FreeImage_Unload(image);
	delete[] pixels;
}

static void loadRAW(const char *filename, std::vector<GLubyte> &data, int width, int height){
	FILE *file = fopen(filename, "rb");
	if (file == NULL){
		perror(filename);
		exit(EXIT_FAILURE);
	}

	fread(&data.front(), sizeof(GLubyte), width * height * 3, file);
	fclose(file);
}

static bool loadImage(const char *filename, std::vector<GLubyte> *data, unsigned int *width, unsigned int *height){
	assert(sizeof(BYTE) == sizeof(GLubyte));

	//image format
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	//pointer to the image, once loaded
	FIBITMAP *dib(0);
	//pointer to the image data
	BYTE* bits(0);
	//image width and height

	//check the file signature and deduce its format
	fif = FreeImage_GetFileType(filename, 0);
	//if still unknown, try to guess the file format from the file extension
	if (fif == FIF_UNKNOWN){
		fif = FreeImage_GetFIFFromFilename(filename);
	}
	//if still unkown, return failure
	if (fif == FIF_UNKNOWN){
		std::cout << "Loading \"" << filename << "\" failure. fif unknown." << std::endl;
		return false;
	}
	// If image foramt is unsupportable, return failure.
	if (fif != FIF_BMP && fif != FIF_JPEG && fif != FIF_GIF && fif != FIF_PNG && fif != FIF_TARGA){
		std::cout << "Loading \"" << filename << "\" failure. unsupportable format." << std::endl;
		return false;
	}

	//check that the plugin has reading capabilities and load the file
	if (FreeImage_FIFSupportsReading(fif))
		dib = FreeImage_Load(fif, filename);
	//if the image failed to load, return failure
	if (!dib){
		std::cout << "Loading \"" << filename << "\" failure. dib error." << std::endl;
		return false;
	}

	// If the format is 8bits, convert it to 32bits.
	unsigned int bytesPerPixel = FreeImage_GetLine(dib) / FreeImage_GetWidth(dib);
	if (bytesPerPixel == 1){
		FIBITMAP *_dib = FreeImage_ConvertTo32Bits(dib);
		FreeImage_Unload(dib);
		dib = _dib;
	}

	unsigned int ff = FreeImage_GetBPP(dib);

	//retrieve the image data
	bits = FreeImage_GetBits(dib);
	//get the image width and height
	*width = FreeImage_GetWidth(dib);
	*height = FreeImage_GetHeight(dib);
	//if this somehow one of these failed (they shouldn't), return failure
	if ((bits == 0) || (*width == 0) || (*height == 0)){
		std::cout << "Loading \"" << filename << "\" failure. error unknown." << std::endl;
		return false;
	}

	data->assign(bits, bits + (*width) * (*height) * bytesPerPixel);

	FreeImage_Unload(dib);

	return true;
}

static void flipImageVertical(std::vector<GLubyte> *data, unsigned int width, unsigned int height, GLenum internalFormat){
	std::vector<GLubyte> temp = *data;
	for (int y = 0; y < height; ++y){
		int offset = 0;
		int offsetDest = 0;
		if (internalFormat == GL_BGR){
			memcpy(&(data->front()) + (height - 1 - y) * 3 * width, &temp.front() + y * 3 * width, 3 * width);
		}
		else if (internalFormat == GL_BGRA){
			memcpy(&(data->front()) + (height - 1 - y) * 4 * width, &temp.front() + y * 4 * width, 4 * width);
		}
	}
}

inline void checkCUDA(cudaError_t e, const char *message = ""){
	if (e != CUDA_SUCCESS){
		std::cout << _cudaGetErrorEnum(e) << ". "<< message << std::endl;
		exit(EXIT_FAILURE);
	}
}

inline void checkCUFFT(cufftResult_t e, const char *message){
	if (e != CUFFT_SUCCESS){
		std::cout << message << std::endl;
		exit(EXIT_FAILURE);
	}
}

class TimerGPU{
private:
	bool enabled;

	cudaEvent_t m_start, m_end;

	float m_minTime, m_maxTime;
	std::vector<float> m_elapsedTimes;

	TimerGPU(TimerGPU const&);
	void operator = (TimerGPU const&);
public:
	TimerGPU()
	:enabled(false), m_minTime(std::numeric_limits<float>::max()), m_maxTime(0.0f){
		checkCUDA(cudaEventCreate(&m_start), "Creating timer event(start).");
		checkCUDA(cudaEventCreate(&m_end), "Creating timer event(stop)");
	}

	~TimerGPU(){
		//checkCUDA(cudaEventDestroy(m_start));
		//checkCUDA(cudaEventDestroy(m_end));
	}

	void enable() {	enabled = true;	}
	void disable(){
		m_elapsedTimes.clear();
		m_minTime = std::numeric_limits<float>::max();
		m_maxTime = 0.0f;
		enabled = false;
	}
	bool isEnabled() const { return enabled; }

	void start(){
		if (enabled){
			checkCUDA(cudaEventRecord(m_start), "Start recoding.");
		}
	}

	float stop(){
		if (enabled){
			checkCUDA(cudaEventRecord(m_end), "End recoding.");
			checkCUDA(cudaEventSynchronize(m_end), "Synchronize timer events.");
			float elapsedTime = 0.0f;
			checkCUDA(cudaEventElapsedTime(&elapsedTime, m_start, m_end), "Getting elapsed time.");
			m_elapsedTimes.push_back(elapsedTime);
			if (m_minTime > elapsedTime){
				m_minTime = elapsedTime;
			}
			if (m_maxTime < elapsedTime){
				m_maxTime = elapsedTime;
			}
			return elapsedTime;
		}
		else{
			return 0.0f;
		}
	}

	void report() const{
		if (m_elapsedTimes.size() > 0){
			std::cout << "Timer report ------" << std::endl;
			std::cout << "MIN : " << m_minTime << std::endl;
			std::cout << "MAX : " << m_maxTime << std::endl;
			std::cout << "AVG : " << getAverageTime() << std::endl;
		}
	}

	float getMinTime() const{ return m_minTime; }
	float getMaxTime() const{ return m_maxTime; }
	float getAverageTime() const{
		float sum = 0.0f;
		for (std::vector<float>::const_iterator it = m_elapsedTimes.begin(); it < m_elapsedTimes.end(); ++it){
			sum += *it;
		}
		return sum / m_elapsedTimes.size();
	}
};