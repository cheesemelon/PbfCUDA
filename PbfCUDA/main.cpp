
#include <iostream>
#include <algorithm>
#include <ctime>
#include <sstream>

#include "shaderBasics.h"
#include "PBFSystem2D.cuh"
#include "Font.h"
#include "FrameRate.h"
#include "BilateralFilterGL.h"
#include "LowpassFilter.cuh"

///////////////////

using namespace std;

void	initRenderingPipeline();
void	render(GLFWwindow* window);

void	reshapeFramebuffer(GLFWwindow* window,  int w, int h);
void	reshapeWindow(GLFWwindow* window,  int w, int h);
void	keyboard(GLFWwindow* window, int key, int scancode, int action, int mods);
void	mouseButton(GLFWwindow* window, int button, int action, int mods);
void	mouseMotion(GLFWwindow* window, double x, double y);

int		FPS	 = 60;

//float	screenScale = 0.4;	// The portion of the screen w.r.t the monitor
float	screenScale = 0.7;	// The portion of the screen w.r.t the monitor
int		screenW, screenH;	// HiDPI
int		windowW, windowH;	// HiDPI

int		numSpritePerWindowWidth = 100;
float	alpha_default = 0.0025 * 10;
float	alpha = alpha_default;

float	fluidAlphaDefault = 0.3;
//float	fluidAlpha = fluidAlphaDefault;
float	fluidAlpha = 1;

float 	scale = 1.5;

float	spriteSize = 0;

// Filter variables
int		kernel_radius = 10;
float	sigma_s = 0.02;
float	sigma_r = 0.1;
int		filter_iteration = 5;
BilateralFilterGL	*filter = NULL;
LowpassFilter		*lowpassFilter = NULL;

Font		*font = NULL;
FrameRate	*frameRate = NULL;

//@ simulation variables
float	WORKSPACE_X = 24;	// 24
float	WORKSPACE_Y = 16;	// 24
float	WORKSPACE_Z = 4;	// 18

PBFSystem2D *pSystem = NULL;
SimulationParameters_f4 param;

bool bWorldRotateY	= false;
bool bGravityOn		= true;	// Currently it works only at the start.
bool bDrawDensity	= false;

float	backgroundDarkness = 1;
bool	debugModeEnabled = false;

glm::vec2 viewport;

// Viewing matrix and vector
glm::vec3 translation;
glm::mat4 scaleMatrix;
glm::mat4 rotationMatrix;
//glm::mat4 modelMatrix;
glm::mat4 viewMatrix;
glm::mat4 projectionMatrix;
glm::mat4 invProjectionMatrix;
glm::mat4 MVPMatrix;

// Viewing control
glm::quat	qX;
glm::quat	qY;
glm::quat	qRot;

glm::vec3	curMouseOnTrackBall;
glm::vec3	preMouseOnTrackBall;
glm::vec3	curMouseOnPlane;
glm::vec3	preMouseOnPlane;
float		trackBallRadius;

float fovy = 45.0f;
float zNear = 10.0f;
float zFar = 70.0f;
//float pplaneWidth	= 0;
//float pplaneHeight	= 0;

//@ camera x-formations
float4 mouseStart;
float4 mouseEnd;
bool bMouseLeftDown = false;
bool bMouseRightDown = false;

//@ depth image type
enum DepthImageTypes
{
	DIMG_ORIGINAL = 0,
	DIMG_BILATERAL_FILTERED,

	NUM_DIMGTYPES
};
DepthImageTypes dImageType = DIMG_ORIGINAL;

//@ render type
enum RenderTypes
{
	RENDER_FLAT = 0,
	RENDER_THICKNESS,
	RENDER_SHADED,
	RENDER_TEST,
	RENDER_TEST_R,
	RENDER_CEL,

	NUM_RENDERTYPES
};
RenderTypes renderType = RENDER_SHADED;

// Rendering
//

glm::vec3 cameraPos(0.0f, 0.0f, 0.5f*(zFar + zNear));
glm::vec3 targetPos(0.0f, 0.0f, 0.0f);

//@ textures
enum Textures
{
	TEX_SPHERE = 0,
	TEX_THICKNESS,
	TEX_DEPTH,
	TEX_DIFFUSE_WRAP,
	TEX_SKYBOX,

	NUM_TEXTURES,
};
std::vector<Texture *> textures(NUM_TEXTURES, NULL);

//@ FBOs
enum FBOS
{
	FBO_DEPTH,
	FBO_FILTER_X,
	FBO_FILTER_Y,

	NUM_FBOS
};
GLuint *pFBOArray = NULL;

//@ VAOs
enum VAOS
{
	VAO_PARTICLE,
	VAO_CUBE,
	VAO_AXIS,
	VAO_FINAL,

	NUM_VAOS
};

GLuint *pVAOArray = NULL;
GLuint particleVbo;
GLuint finalVbo;

//@ quad for rendering
float2 quadVert[] = {
	make_float2(-1.0f, -1.0f),
	make_float2( 1.0f, -1.0f),
	make_float2( 1.0f,  1.0f),
	make_float2(-1.0f,  1.0f),
};

//@ shaders
enum ShaderType
{
	SHDR_DEPTH = 0,
	SHDR_THICKNESS,
	SHDR_TEST,
	SHDR_SKYBOX,

	NUM_SHADERS
};
std::vector<Shader *> shaders(NUM_SHADERS, NULL);
const char *shader_filenames[NUM_SHADERS][2] = {
	{ "depth.vert", "depth.frag" },
	{ "thickness.vert", "thickness.frag" },
	{ "test.vert", "test.frag" },
	{ "skybox.vert", "skybox.frag" },
};

// Drop the particles
void
DamBreaking()
{
	// Initial velocity
	float4 vel = make_float4(0, 0, 0, 0);
	float4 pos = make_float4(-0.9998*0.5*WORKSPACE_X, -0.9998*0.5*WORKSPACE_Y, -0.9998*0.5*WORKSPACE_Z, 0);
	float spacing = 0.62f;
	
	for(int i = 0; i < 60; i++)
	{
		pSystem->createParticleNbyN(
		//	40,
			15,
			spacing,
			pos + make_float4(0, i*spacing*pSystem->hSimParams_f.h, 0, 0),
			vel);
	}
}

int
main(int argc, char *argv[])
{
	// Initialize the OpenGL system
	GLFWwindow*	window = initOpenGL(argc, argv, screenScale, screenW, screenH, windowW, windowH);
	if (window == NULL) return  -1;

	glfwSetWindowTitle(window, "PBF with CUDA");

	reshapeWindow(window, screenW, screenH);
	reshapeFramebuffer(window, windowW, windowH);

	// Initialize the simulation system
	param.workspace = make_float4(WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z, 0);
	param.gravity = make_float4(0, (bGravityOn ? -9.8 : 0), 0, 0);

	pSystem = new PBFSystem2D();
	pSystem->initParticleSystem(param);

	// Initialize the rendering pipeline
	initRenderingPipeline();

	frameRate = FrameRate::create();

	// Drop the particles
	DamBreaking();

	pSystem->simulate(1.0 / FPS);
	pSystem->bPause = true;

	// Add callbacks
	glfwSetWindowSizeCallback(window, reshapeWindow);
	glfwSetFramebufferSizeCallback(window, reshapeFramebuffer);
	glfwSetKeyCallback(window, keyboard);
	glfwSetMouseButtonCallback(window, mouseButton);
	glfwSetCursorPosCallback(window, mouseMotion);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		// Record elapsed time.
		frameRate->elapse();

		// Window title with the current number of particles
		char winTitle[256] = {0,};
		sprintf(winTitle, "PBF with %d particles, %6.2f FPS", pSystem->numParticles, frameRate->getFPS());
		glfwSetWindowTitle(window, winTitle);

		// Simulation and then render
		render(window);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Free resources
	for (auto &shader : shaders){
		delete shader;
	}
	for (auto &texture : textures){
		delete texture;
	}
	delete filter, lowpassFilter;
	delete font, frameRate;
	delete pSystem;

	glfwTerminate();
	return	0;
}

void
reshapeWindow(GLFWwindow* window, int w, int h )
{
	screenW = w;
	screenH = h;

	trackBallRadius = (screenW > screenH) ? 0.5f*screenW: 0.5f*screenH;

//	cout << "The window size changed into " << w << " x " << h << endl;
}

void
reshapeFramebuffer(GLFWwindow* window, int w, int h )
{
	windowW = w;
	windowH = h;

	viewport = glm::vec2((float)windowW, (float)windowH);

	float	aspect = (float)w/h;
	projectionMatrix	= glm::perspective(fovy, aspect, zNear, zFar);
	invProjectionMatrix = glm::inverse(projectionMatrix);

	glViewport( 0, 0, w, h );

//	cout << "The framebuffer size changed into " << w << " x " << h << " with aspect ratio = " << aspect << endl;
}

void
mouseButton(GLFWwindow* window, int button, int action, int mods)
{
	double	x, y;
	glfwGetCursorPos(window, &x, &y);

	float convertedX = (x - 0.5*screenW);
	float convertedY = -(y - 0.5*screenH);

	if(convertedX >= 0.5*screenW)	convertedX = 0.5*screenW;
	if(convertedX <= -0.5*screenW)	convertedX = -0.5*screenW;
	if(convertedY >= 0.5*screenH)	convertedY = 0.5*screenH;
	if(convertedY <= -0.5*screenH)	convertedY = -0.5*screenH;

	switch(button)
	{
	case GLFW_MOUSE_BUTTON_LEFT:
		if (action == GLFW_PRESS)
		{
			bMouseLeftDown = true;

			glm::vec3 vMax(convertedX, convertedY, 0);
			if(glm::length(vMax) > trackBallRadius)
				vMax = trackBallRadius*glm::normalize(vMax);

			float zSquared = trackBallRadius*trackBallRadius - vMax.x*vMax.x - vMax.y*vMax.y;
			if(zSquared < 0) zSquared = 0;
			float z = sqrt(zSquared);

			curMouseOnTrackBall = glm::normalize(glm::vec3(vMax.x, vMax.y, z ));
			preMouseOnTrackBall = curMouseOnTrackBall;
		}
		else	{
			bMouseLeftDown = false;
		}
		break;

	case GLFW_MOUSE_BUTTON_RIGHT:
		if (action == GLFW_PRESS)
		{
			bMouseRightDown = true;
			
			curMouseOnPlane = glm::vec3(convertedX, convertedY, 0);
			preMouseOnPlane = curMouseOnPlane;
		}
		else	{
			bMouseRightDown = false;
		}
		break;
	}
}

void
mouseMotion(GLFWwindow* window, double x, double y)
{
	float convertedX =  (x - 0.5*screenW);
	float convertedY = -(y - 0.5*screenH);

	if(convertedX >=  0.5*screenW)	convertedX =  0.5*screenW;
	if(convertedX <= -0.5*screenW)	convertedX = -0.5*screenW;
	if(convertedY >=  0.5*screenH)	convertedY =  0.5*screenH;
	if(convertedY <= -0.5*screenH)	convertedY = -0.5*screenH;

	if(bMouseLeftDown)
	{
		glm::vec3 axisX(1.0f, 0.0f, 0.0f);
		glm::vec3 axisY(0.0f, 1.0f, 0.0f);

		glm::vec3 vMax(convertedX, convertedY, 0);
		if(glm::length(vMax) > trackBallRadius)
			vMax = trackBallRadius*glm::normalize(vMax);

		float zSquared = trackBallRadius*trackBallRadius - vMax.x*vMax.x - vMax.y*vMax.y;
		if(zSquared < 0) zSquared = 0;
		float z = sqrt(zSquared);

		curMouseOnTrackBall = glm::vec3(vMax.x, vMax.y, z );
		curMouseOnTrackBall = glm::normalize(curMouseOnTrackBall);

		float k_rot = 1.5;
		// Y
		glm::vec3 newTrack = glm::dot(curMouseOnTrackBall - preMouseOnTrackBall, axisX)*axisX;
		glm::vec3 xCur = glm::normalize(preMouseOnTrackBall + newTrack);

		glm::vec3 axis = glm::cross(preMouseOnTrackBall, xCur);
		float sinTheta = k_rot*glm::length(axis);
		axis = (xCur.x - preMouseOnTrackBall.x > 0 ? 1.0f : -1.0f)*sin(0.5f*sinTheta)*axisY;
	
		glm::quat q(cos(0.5*sinTheta), axis);
		qY = glm::normalize(qY*q);

		// X
		newTrack = glm::dot(curMouseOnTrackBall - preMouseOnTrackBall, axisY)*axisY;
		xCur = glm::normalize(preMouseOnTrackBall + newTrack);

		axis = glm::cross(preMouseOnTrackBall, xCur);
		sinTheta = k_rot*glm::length(axis);
		axis = (xCur.y - preMouseOnTrackBall.y > 0 ? -1.0f : 1.0f)*sin(0.5f*sinTheta)*axisX;

		q = glm::quat(cos(0.5*sinTheta), axis);
		qX = glm::normalize(qX*q);

		rotationMatrix = glm::mat4_cast(qX*qY);
	}

	else if(bMouseRightDown)
	{
		curMouseOnPlane = glm::vec3(convertedX, convertedY, 0);
		translation = translation + curMouseOnPlane - preMouseOnPlane;
	}

	preMouseOnTrackBall = curMouseOnTrackBall;
	preMouseOnPlane		= curMouseOnPlane;
}

void
initRenderingPipeline()
{
	// Rotation matrix
	float angleX = 10;	// -10 or 0
	float angleY = 0;

	qX = glm::quat(cos(0.5f*angleX*glm::pi<float>()/180.0f), sin(0.5f*angleX*glm::pi<float>()/180.0f ), 0.0f, 0.0f);
	qY = glm::quat(cos(0.5f*angleY*glm::pi<float>()/180.0f), 0.0f, sin(0.5f*angleY*glm::pi<float>()/180.0f ), 0.0f);
	qRot = qX * qY;
	rotationMatrix = glm::mat4_cast(qRot);

	// Scale matrix
	scaleMatrix = glm::mat4(glm::mat3(scale));
	
	// View matrix
	viewMatrix = rotationMatrix * scaleMatrix;

	// Viewing control with mouse
	bMouseLeftDown = false;
	bMouseRightDown = false;

	trackBallRadius = 0.5f*screenW;
	curMouseOnTrackBall = glm::vec3(0.0f);
	preMouseOnTrackBall = glm::vec3(0.0f);
	curMouseOnPlane = glm::vec3(0.0f);
	preMouseOnPlane = glm::vec3(0.0f);
	
	// Point sprite is deprecated.
	//if (0)
	//{
	//	//@ Prepare sphere texture for fluid rendering
	//	glGenTextures(1, &textureIds[TEX_SPHERE]);

	//	glBindTexture(GL_TEXTURE_2D, textureIds[TEX_SPHERE]);
	//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	//	checkOpenGL("glTexImage2D", __FILE__, __LINE__, false, true);

	//	glEnable(GL_POINT_SPRITE);
	//	checkOpenGL("glEnable", __FILE__, __LINE__, false, true);
	//	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	//	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	//	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	//	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//}

	if (pFBOArray) delete[] pFBOArray;
	pFBOArray = new GLuint[NUM_FBOS];
	if (pVAOArray) delete[] pVAOArray;
	pVAOArray = new GLuint[NUM_VAOS];

	//@ create VAO
	glGenVertexArrays(NUM_VAOS, pVAOArray);

	//@ create VBO
	glBindVertexArray(pVAOArray[VAO_PARTICLE]);
	glGenBuffers(1, &particleVbo);

	glBindVertexArray(pVAOArray[VAO_FINAL]);
	glGenBuffers(1, &finalVbo);
	glBindBuffer(GL_ARRAY_BUFFER, finalVbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * 4, quadVert, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

//	pplaneHeight	= 2.0f*zNear*tan(0.5f*fovy/180.0f*glm::pi<float>());
//	pplaneWidth		= pplaneHeight*aspect;

	// Load shaders
	for (int i = 0; i < NUM_SHADERS; ++i){
		delete shaders[i];
		shaders[i] = Shader::createWithFile(shader_filenames[i][0], shader_filenames[i][1]);
	}

	// Variouse textures
	//
	// create thickness texture
	delete textures[TEX_THICKNESS];
	textures[TEX_THICKNESS] = Texture2D::createEmpty(GL_R32F, windowW, windowH, GL_RED, GL_FLOAT);
	textures[TEX_THICKNESS]->setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
	textures[TEX_THICKNESS]->setFilter(GL_LINEAR, GL_LINEAR);

	// create depth texture
	delete textures[TEX_DEPTH];
	textures[TEX_DEPTH] = Texture2D::createEmpty(GL_DEPTH_COMPONENT32F, windowW, windowH, GL_DEPTH_COMPONENT, GL_FLOAT);
	textures[TEX_DEPTH]->setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
	textures[TEX_DEPTH]->setFilter(GL_LINEAR, GL_LINEAR);

	// create skybox texture
	delete textures[TEX_SKYBOX];
	textures[TEX_SKYBOX] = TextureCubemap::createWithFiles("resources/TropicalSunnyDay/TropicalSunnyDayLeft2048.png",
														"resources/TropicalSunnyDay/TropicalSunnyDayRight2048.png",
														"resources/TropicalSunnyDay/TropicalSunnyDayUp2048.png",
														"resources/TropicalSunnyDay/TropicalSunnyDayDown2048.png",
														"resources/TropicalSunnyDay/TropicalSunnyDayFront2048.png",
														"resources/TropicalSunnyDay/TropicalSunnyDayBack2048.png");
	textures[TEX_SKYBOX]->setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
	textures[TEX_SKYBOX]->setFilter(GL_LINEAR, GL_LINEAR);

	// create diffuse wrap texture
	delete textures[TEX_DIFFUSE_WRAP];
	textures[TEX_DIFFUSE_WRAP] = Texture1D::create(ImageData::createWithRawFile("resources/diffuse wrap_spirit of sea.raw", 256, 1, 3));
	textures[TEX_DIFFUSE_WRAP]->setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
	textures[TEX_DIFFUSE_WRAP]->setFilter(GL_LINEAR, GL_LINEAR);

	//@ create FBO_DEPTH
	glGenFramebuffers(1, &pFBOArray[FBO_DEPTH]);
	glBindFramebuffer(GL_FRAMEBUFFER, pFBOArray[FBO_DEPTH]);
	{
		//@ attach color textures
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[TEX_THICKNESS]->getID(), 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, textures[TEX_DEPTH]->getID(), 0);

		//@ set the list of draw buffers
		GLuint attachments[] = { GL_COLOR_ATTACHMENT0 };
		glDrawBuffers(sizeof(attachments) / sizeof(GLuint), attachments);
		checkOpenGL("glDrawBuffers()", __FILE__, __LINE__, false, true);

		//@ check framebuffer
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		{
			cout << "Error in FBO_DEPTH." << endl;
			exit(1);
		}
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Delete and create separable bilateral filter for depth smoothing.
	spriteSize = (float)screenH / numSpritePerWindowWidth * scale;
	delete filter;
	filter = BilateralFilterGL::create(windowW, windowH, kernel_radius, spriteSize * sigma_s, spriteSize * sigma_r, filter_iteration, projectionMatrix, invProjectionMatrix);

	// Delete and create lowpass filter for thickness smoothig.
	delete lowpassFilter;
	lowpassFilter = LowpassFilter::create(windowW, windowH, 20.0f, true);
	lowpassFilter->setTexture(textures[TEX_THICKNESS]->getID());

	// Delete and create text renderder. This is bad module(especially on performance) and should be changed.
	delete font;
	font = Font::create(finalVbo);
}

void
render(GLFWwindow* window)
{
	if(!pSystem->bPause)	pSystem->simulate(1.0 / FPS);

	// Check the number of particls
	if (pSystem->numParticles == 0)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		cout << "# particles = 0" << endl;

		return;
	}

	if(bWorldRotateY)
	{
		float angleY = 0.5;

		glm::quat q(cos(0.5f*angleY/180.0f*glm::pi<float>()), 0.0f, sin(0.5f*angleY/180.0f*glm::pi<float>()), 0.0f);
		qY = glm::normalize(qY*q);

		rotationMatrix	= glm::mat4_cast(qX*qY);
	}

	//@ intermediate scene
	//@ set MVP matrix
	viewMatrix	= glm::lookAt(cameraPos, targetPos, glm::vec3(0, 1.0f, 0));
	viewMatrix	= viewMatrix * rotationMatrix * scaleMatrix;
	MVPMatrix	= projectionMatrix * viewMatrix;

	// Depth and Thickness
	//
	glBindFramebuffer(GL_FRAMEBUFFER, pFBOArray[FBO_DEPTH]);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	{
		// Copy particle data to array buffer.
		//
		glBindBuffer(GL_ARRAY_BUFFER, particleVbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*pSystem->numParticles, pSystem->h_p, GL_STREAM_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		spriteSize = (float)screenH / numSpritePerWindowWidth * scale;

		// Depth of the fluid surface
		//
		shaders[SHDR_DEPTH]->bind();
		{
			shaders[SHDR_DEPTH]->setUniform("spriteSize", spriteSize);
			shaders[SHDR_DEPTH]->setUniform("V", viewMatrix);
			shaders[SHDR_DEPTH]->setUniform("P", projectionMatrix);
			//glUniformMatrix4fv(UNILOC_DEPTH::anisotropyMatLoc, 1, GL_FALSE, &pSystem->h_mat[0].x); 

			glEnable(GL_DEPTH_TEST);	// Enable the depth test to obtain the eye-closest surface
			glDrawArrays(GL_POINTS, 0, pSystem->numParticles);
			glDisable(GL_DEPTH_TEST);	// Disable the depth test to accumulate the thickness
		}
		Shader::bindDefault();
		checkOpenGL("Depth Shader", __FILE__, __LINE__, false, true);

		// Thickness of the fluid to mimic volume of fluid
		//
		shaders[SHDR_THICKNESS]->bind();
		{
			shaders[SHDR_THICKNESS]->setUniform("spriteSize", spriteSize);
			shaders[SHDR_THICKNESS]->setUniform("alpha", alpha);
			shaders[SHDR_THICKNESS]->setUniform("MVP", MVPMatrix);

			glEnable(GL_BLEND);			// Accumlate the thickness using the additive alpha blending
			glBlendFunc(GL_ONE, GL_ONE);
			glDrawArrays(GL_POINTS, 0, pSystem->numParticles);
			glDisable(GL_BLEND);

			Texture2D::bindDefault();
		}
		Shader::bindDefault();
		checkOpenGL("Thickness Shader", __FILE__, __LINE__, false, true);

		glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glDisableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		if (dImageType == DIMG_BILATERAL_FILTERED){
			// Filter on depth.
			filter->run(textures[TEX_DEPTH]->getID());
		}
		// Filter on thickness.
		lowpassFilter->run();
	}

	//@ final scene
	//
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//	float scaledZNear = zNear*((float)windowW/pplaneWidth);
	{
		// Render skybox
		shaders[SHDR_SKYBOX]->bind();
		{
			shaders[SHDR_SKYBOX]->setUniform("invV", glm::inverse(glm::mat3(viewMatrix)));
			shaders[SHDR_SKYBOX]->setUniform("invP", invProjectionMatrix);

			glActiveTexture(GL_TEXTURE0);
			textures[TEX_SKYBOX]->bind();

			glBindBuffer(GL_ARRAY_BUFFER, finalVbo);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// GL_QUADS is dprecated.

			glDisableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			TextureCubemap::bindDefault();
		}
		Shader::bindDefault();
		checkOpenGL("Skybox Shader.", __FILE__, __LINE__, false, true);

		shaders[SHDR_TEST]->bind();
		{
			shaders[SHDR_TEST]->setUniform("viewport", viewport);
			shaders[SHDR_TEST]->setUniform("alpha", fluidAlpha);
			shaders[SHDR_TEST]->setUniform("renderType", renderType);
			shaders[SHDR_TEST]->setUniform("projectionMatrix", projectionMatrix);
			shaders[SHDR_TEST]->setUniform("invProjectionMatrix", invProjectionMatrix);

			glActiveTexture(GL_TEXTURE0);
			textures[TEX_DEPTH]->bind();

			glActiveTexture(GL_TEXTURE1);
			textures[TEX_THICKNESS]->bind();

			glActiveTexture(GL_TEXTURE2);
			textures[TEX_DIFFUSE_WRAP]->bind();

			glActiveTexture(GL_TEXTURE3);
			textures[TEX_SKYBOX]->bind();

			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, finalVbo);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// GL_QUADS is dprecated.
			glDisable(GL_BLEND);

			glDisableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			Texture1D::bindDefault();
			Texture2D::bindDefault();
			TextureCubemap::bindDefault();
		}
		Shader::bindDefault();
		checkOpenGL("Final Render Shader", __FILE__, __LINE__, false, true);
	}
	
	// Render text.
	//
	// Make status text.
	std::stringstream ss;
	ss << "< STATUS > " << frameRate->getMPF() << "mpf" << std::endl;
	std::string str;
	switch (renderType)
	{
	case RenderTypes::RENDER_TEST:			str = "Test Shading";		break;
	case RenderTypes::RENDER_CEL:			str = "Cel Shading";		break;
	case RenderTypes::RENDER_TEST_R:		str = "Test Shading + Reflection + Refraction";	break;
	case RenderTypes::RENDER_FLAT:			str = "Depth";				break;
	case RenderTypes::RENDER_SHADED:		str = "Shaded";				break;
	case RenderTypes::RENDER_THICKNESS:		str = "Thickness";			break;
	default:								str = "";					break;
	}
	ss << "Rendering mode : " << str << std::endl;
	switch (dImageType)
	{
	case DepthImageTypes::DIMG_BILATERAL_FILTERED:		str = "Bilateral filtered";		break;
	case DepthImageTypes::DIMG_ORIGINAL:				str = "Original";				break;
	default:											str = "";						break;
	}
	ss << "Depth Image : " << str << std::endl;
	ss << "Kernel Radius : " << kernel_radius << std::endl;
	ss << "Sigma S : " << sigma_s << std::endl;
	ss << "Sigma R : " << sigma_r << std::endl;
	ss << "# of Filter Iteration : " << filter_iteration << std::endl;
	ss << "3D Filtering : " << (filter->b3DFiltering ? "ON" : "OFF") << std::endl;
	ss << "Depth Correction : " << (filter->bDepthCorrection ? "ON" : "OFF") << std::endl;

	// Draw text twice for outline.
	font->setStyle(FONT_STYLE_BOLD);
	font->setColor(0.0f, 0.0f, 0.0f, 1.0f);
	font->drawText(0, 0, ss.str().c_str());
	font->setStyle(FONT_STYLE_REGULAR);
	font->setColor(1.0f, 1.0f, 1.0f, 1.0f);
	font->drawText(0, 0, ss.str().c_str());
}

void
keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// Control filter parameters
	if (mods == GLFW_MOD_CONTROL){
		switch (key)
		{
		// Change kernel radius
		case GLFW_KEY_K:
			if (action == GLFW_PRESS){
				kernel_radius += 5.0f;
				if (kernel_radius > 50.0f){
					kernel_radius = 5.0f;
				}
			}
			break;
		// Change sigma_s
		case GLFW_KEY_S:
			if (action == GLFW_PRESS){
				sigma_s += 0.01;
				if (sigma_s > 0.2){
					sigma_s = 0.01;
				}
			}
			break;
		// Change sigma_r
		case GLFW_KEY_R:
			if (action == GLFW_PRESS){
				sigma_r += 0.05;
				if (sigma_r > 1.0){
					sigma_r = 0.05;
				}
			}
			break;
		// Change # of filter iterations
		case GLFW_KEY_I:
			if (action == GLFW_PRESS){
				++filter_iteration;
				if (filter_iteration > 10){
					filter_iteration = 1;
				}
			}
			break;
		case GLFW_KEY_LEFT_BRACKET:
			if (action == GLFW_PRESS){
				filter->toggle3DFiltering();
			}
			break;
		case GLFW_KEY_RIGHT_BRACKET:
			if (action == GLFW_PRESS){
				filter->toggleDepthCorrection();
			}
			break;
		default:
			break;
		}
		
		float _sigma_s;
		if (filter->b3DFiltering){
			_sigma_s = spriteSize * sigma_s;
		}
		else{
			// Set sigma_s as a quater of kernel_radius.
			_sigma_s = 0.25f * kernel_radius;
		}
		float _sigma_r;
		if (filter->bDepthCorrection){
			_sigma_r = spriteSize * sigma_r;
		}
		else{
			// Range should be in [0.0 1.0]
			_sigma_r = sigma_r;
		}
		filter->setParameters(kernel_radius, _sigma_s, _sigma_r, filter_iteration);
		return;
	}

	switch (key)
	{
		// Toggle play/pause
	case GLFW_KEY_P:
	case GLFW_KEY_SPACE:
		if (action == GLFW_PRESS)
		{
			pSystem->bPause = !pSystem->bPause;
		}
		break;

		// Restart the simulation
	case GLFW_KEY_I:
		if (action == GLFW_PRESS)
		{
			// Initialize the simulation system
			param.workspace = make_float4(WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z, 0);
			param.gravity = make_float4(0, (bGravityOn ? -9.8 : 0), 0, 0);

			if (pSystem)	delete pSystem;

			pSystem = new PBFSystem2D();
			pSystem->initParticleSystem(param);

			delete frameRate;
			frameRate = FrameRate::create();

			// Initialize the rendering pipeline
			initRenderingPipeline();

			// Drop the particles
			DamBreaking();
		}
		break;

		// Toggle the smoothing mode on/off
	case GLFW_KEY_B:
		if (action == GLFW_PRESS)
		{
			dImageType = (DepthImageTypes)((dImageType + 1) % NUM_DIMGTYPES);
		}
		break;

		// Change the shading method
	case GLFW_KEY_S:
		if (action == GLFW_PRESS)
		{
			renderType = (RenderTypes)((renderType + 1) % NUM_RENDERTYPES);
		}
		break;

		// Toggle the debug mode on/off
	case GLFW_KEY_D:
		if (action == GLFW_PRESS)
		{
			debugModeEnabled = !debugModeEnabled;
		}
		break;

		// Increase the background lightness
	case GLFW_KEY_L:
		if (action == GLFW_PRESS)
		{
			backgroundDarkness += 0.25;
			if (backgroundDarkness > 1)	backgroundDarkness = 0;
		}
		break;

		// Toggle the transparency
	case GLFW_KEY_T:
		if (action == GLFW_PRESS)
		{
			fluidAlpha = (fluidAlpha == fluidAlphaDefault) ? 1.0 : fluidAlphaDefault;
		}
		break;

	case GLFW_KEY_G:
		if (action == GLFW_PRESS)
		{
			bGravityOn = !bGravityOn;
		}
		break;
	case GLFW_KEY_R:
		if (action == GLFW_PRESS)
		{
			bWorldRotateY = !bWorldRotateY;
		}
		break;

		// Control the point sprite size
		//
	case GLFW_KEY_MINUS: // -: Decrease the sprite size
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
		{
			numSpritePerWindowWidth++;
		}
		break;

	case GLFW_KEY_EQUAL: // +: Increase the sprite size
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
		{
			numSpritePerWindowWidth = max(numSpritePerWindowWidth - 1, 10);
		}
		break;

	case GLFW_KEY_1:	// Predefined sprite size
		numSpritePerWindowWidth = 100;
		alpha = alpha_default;
		break;

	case GLFW_KEY_2:	// Predefined sprite size
		numSpritePerWindowWidth = 150;
		alpha = pow(1.5, 2) * alpha_default;
		break;

	case GLFW_KEY_3:	// Predefined sprite size
		numSpritePerWindowWidth = 200;
		alpha = pow(2.0, 2) * alpha_default;
		break;

	case GLFW_KEY_4:	// Predefined sprite size
		numSpritePerWindowWidth = 250;
		alpha = pow(2.5, 2) * alpha_default;
		break;

		// Zoom in
	case GLFW_KEY_Z:
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
		{
			scale += 0.1f;
			scaleMatrix = glm::mat4(glm::mat3(scale));
		}
		break;

		// Zoom out
	case GLFW_KEY_X:
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
		{
			scale = max(scale - 0.1f, 0.1f);
			scaleMatrix = glm::mat4(glm::mat3(scale));
		}
		break;

		// ESC
	case GLFW_KEY_ESCAPE:
		glfwSetWindowShouldClose(window, GL_TRUE);
		break;

		// Others
	default:
		cout << "Unsupported key = " << key << endl;
		break;
	}
}