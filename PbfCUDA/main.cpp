#include <iostream>
#include <algorithm>
#include <ctime>
#include <sstream>

#include "shaderBasics.h"
#include "PBFSystem2D.cuh"
#include "BilateralFilter.cuh"
#include "SFBF.cuh"
#include "temp.h"
#include "Font.h"
#include "BFGL.h"

// temp
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
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

//@ Filter
float	freq_radius = 10.0f;
int		sigma_r_SFBF = 40;
int		sigma_r = 5;
int		sigma_s = 5;
int		iterations = 5;
int		dog_sigma = 1;
int		dog_radius = 20;
float		dog_similarity = 0.99f;
std::vector<GLubyte> texdata_diffuseWrap(256 * 3);
unsigned int skybox_width, skybox_height;
std::vector<GLubyte> texdata_skybox[6];
std::vector<GLfloat> screenshot1, screenshot2;
BilateralFilter *filter = NULL;
Font	*font = NULL;
TimerGPU *timer1, *timer2 = NULL;
int genNormal = 0;
int restorePos = 0;
BFGL *filter_BFGL = NULL;

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

	// temp
	loadRAW("resources/diffuse wrap_spirit of sea.raw", texdata_diffuseWrap, 256, 1);
	loadImage("Resources/iceflats/iceflats_rt.tga", &texdata_skybox[0], &skybox_width, &skybox_height);
	loadImage("Resources/iceflats/iceflats_lf.tga", &texdata_skybox[1], &skybox_width, &skybox_height);
	loadImage("Resources/iceflats/iceflats_up.tga", &texdata_skybox[2], &skybox_width, &skybox_height);
	loadImage("Resources/iceflats/iceflats_dn.tga", &texdata_skybox[3], &skybox_width, &skybox_height);
	loadImage("Resources/iceflats/iceflats_ft.tga", &texdata_skybox[4], &skybox_width, &skybox_height);
	loadImage("Resources/iceflats/iceflats_bk.tga", &texdata_skybox[5], &skybox_width, &skybox_height);
	//loadImage("Resources/TropicalSunnyDay/TropicalSunnyDayLeft2048.png", &texdata_skybox[0], &skybox_width, &skybox_height);
	//loadImage("Resources/TropicalSunnyDay/TropicalSunnyDayRight2048.png", &texdata_skybox[1], &skybox_width, &skybox_height);
	//loadImage("Resources/TropicalSunnyDay/TropicalSunnyDayUp2048.png", &texdata_skybox[2], &skybox_width, &skybox_height);
	//loadImage("Resources/TropicalSunnyDay/TropicalSunnyDayDown2048.png", &texdata_skybox[3], &skybox_width, &skybox_height);
	//loadImage("Resources/TropicalSunnyDay/TropicalSunnyDayFront2048.png", &texdata_skybox[4], &skybox_width, &skybox_height);
	//loadImage("Resources/TropicalSunnyDay/TropicalSunnyDayBack2048.png", &texdata_skybox[5], &skybox_width, &skybox_height);
	for (int i = 0; i < 6; ++i){
		flipImageVertical(&texdata_skybox[i], skybox_width, skybox_height, GL_BGR);
	}

	reshapeWindow(window, screenW, screenH);
	reshapeFramebuffer(window, windowW, windowH);

	// Initialize the simulation system
	param.workspace = make_float4(WORKSPACE_X, WORKSPACE_Y, WORKSPACE_Z, 0);
	param.gravity = make_float4(0, (bGravityOn ? -9.8 : 0), 0, 0);

	pSystem = new PBFSystem2D();
	pSystem->initParticleSystem(param);

	// Initialize the rendering pipeline
	font = new Font();
	initRenderingPipeline();

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

	// Time
	double	t = 0, t_old = 0;

	const int	NUM_HISTORY = 30;

	double	history[NUM_HISTORY];
	int		index = 0;
	for (int i = 0; i < NUM_HISTORY; i++)
			history[i] = 0;
	double descrete_fps = 0.0;

	glfwSetTime(t);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		// Window title with the current number of particles
		char winTitle[256] = {0,};
		sprintf(winTitle, "PBF with %d particles", pSystem->numParticles );

		//if (1)
		//{
			t = glfwGetTime();
			double	dt = t - t_old;
			t_old = t;

			int	num_elapsed = min(index + 1, NUM_HISTORY);
			history[index % NUM_HISTORY] = dt * 1000;

			double	average_dt = 0;
			for (int i = 0; i < num_elapsed; i++)
				average_dt += history[i];	
			average_dt /= num_elapsed;

			sprintf(winTitle, "%s, %6.2f FPS", winTitle, 1000 / average_dt);
			//cout << winTitle << endl;

			index++;
		//}

		glfwSetWindowTitle(window, winTitle);

		// Simulation and then render
		render(window);

		if (index % 5 == 0){
			descrete_fps = 1 / dt;
		}
		//font->drawText(200, 0, "FPS %.2lf, %lf", descrete_fps, dt);
		font->drawText(200, 0, "FPS %.2f", 1000.0 / average_dt);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	delete filter;
	delete filter_BFGL;
	timer1->report();
	delete timer1;
	timer2->report();
	delete timer2;
	delete font;
	delete pSystem;

	glfwTerminate();
	return	0;
}

glm::vec2 viewport;

// Viewing matrix and vector
glm::vec3 translation;
glm::mat4 scaleMatrix;
glm::mat4 rotationMatrix;
//glm::mat4 modelMatrix;
glm::mat4 viewMatrix;
glm::mat4 projectionMatrix;
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

float fovy		= 45.0f;
float zNear		= 10.0f;
float zFar		= 70.0f;
//float pplaneWidth	= 0;
//float pplaneHeight	= 0;

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

	glViewport( 0, 0, w, h );

//	cout << "The framebuffer size changed into " << w << " x " << h << " with aspect ratio = " << aspect << endl;
}

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
	DIMG_SEPERABLE_FILTERED,

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
	RENDER_OUTLINE,
	RENDER_CEL,

	NUM_RENDERTYPES
};

RenderTypes renderType = RENDER_THICKNESS;

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

// Rendering
//

glm::vec3 cameraPos(0.0f, 0.0f, 0.5f*(zFar + zNear));
glm::vec3 targetPos(0.0f, 0.0f, 0.0f);


//@ textures
enum Textures
{
	TEX_SPHERE = 0,
	TEX_COLOR,
	TEX_THICKNESS,
	TEX_DEPTH,
	TEX_DIFFUSE_WRAP,
	TEX_OUTLINE,
	TEX_SKYBOX,

	NUM_TEXTURES,
};

GLuint textureIds[NUM_TEXTURES];
GLuint renderbuffer_depth;

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
GLuint finalVbo[2];

//@ quad for rendering
float2 quadVert[] = {
	make_float2(-1.0f, -1.0f),
	make_float2( 1.0f, -1.0f),
	make_float2( 1.0f,  1.0f),
	make_float2(-1.0f,  1.0f),
};
float2 texCoord[4] = {
	make_float2(0.0f, 0.0f),
	make_float2(1.0f, 0.0f),
	make_float2(1.0f, 1.0f),
	make_float2(0.0f, 1.0f),
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

char* shaders[][2] = {
	{(char*)"depth.vert",		(char*)"depth.frag"},
	{(char*)"thickness.vert",	(char*)"thickness.frag"},
	{(char*)"test.vert",		(char*)"test.frag"},
	{ (char*)"skybox.vert", (char *)"skybox.frag" },
};

GLuint programIds[10] = {0,};

//@ uniform location
namespace UNILOC_DEPTH
{
	GLuint projMatLoc		= 0;
	GLuint viewMatLoc		= 0;
	GLuint MVPMatLoc		= 0;
	GLuint viewportLoc		= 0;
	GLuint spriteSizeLoc	= 0; 
	GLuint alphaLoc			= 0;
};

namespace UNILOC_THICKNESS
{
	GLuint projMatLoc		= 0;
	GLuint viewMatLoc		= 0;
	GLuint MVPMatLoc		= 0;
	GLuint viewportLoc		= 0;
	GLuint spriteSizeLoc	= 0; 
	GLuint alphaLoc         = 0; 
};

namespace UNILOC_TEST
{
	GLuint projMatLoc		= 0;
	GLuint invProjMatLoc	= 0;
	GLuint modelViewMatLoc	= 0;
	GLuint viewportLoc		= 0;
	GLuint zNearLoc			= 0;
	GLuint renderType		= 0;
	GLuint alphaLoc			= 0;
};

namespace UNILOC_SKYBOX
{
	GLuint viewMatLoc = 0;
	GLuint projMatLoc = 0;
};

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
	if (0)
	{
		//@ Prepare sphere texture for fluid rendering
		glGenTextures(1, &textureIds[TEX_SPHERE]);

		glBindTexture(GL_TEXTURE_2D, textureIds[TEX_SPHERE]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		checkOpenGL("glTexImage2D", __FILE__, __LINE__, false, true);

		glEnable(GL_POINT_SPRITE);
		checkOpenGL("glEnable", __FILE__, __LINE__, false, true);
		glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	//@ loading shaders
	for( int i = 0; i < NUM_SHADERS; i++ )
	{
		programIds[i] = setShaders(shaders[i][0], shaders[i][1]);
	}

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
	glGenBuffers(1, finalVbo);
	glBindBuffer(GL_ARRAY_BUFFER, finalVbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * 4, quadVert, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

//	pplaneHeight	= 2.0f*zNear*tan(0.5f*fovy/180.0f*glm::pi<float>());
//	pplaneWidth		= pplaneHeight*aspect;

	//@ get uniform location
	UNILOC_DEPTH::projMatLoc	= glGetUniformLocation(programIds[SHDR_DEPTH], "P");
	UNILOC_DEPTH::viewMatLoc	= glGetUniformLocation(programIds[SHDR_DEPTH], "V");
	UNILOC_DEPTH::MVPMatLoc		= glGetUniformLocation(programIds[SHDR_DEPTH], "MVP");
	UNILOC_DEPTH::viewportLoc	= glGetUniformLocation(programIds[SHDR_DEPTH], "viewport");
	UNILOC_DEPTH::spriteSizeLoc = glGetUniformLocation(programIds[SHDR_DEPTH], "spriteSize");
	UNILOC_DEPTH::alphaLoc		= glGetUniformLocation(programIds[SHDR_DEPTH], "alpha");

	UNILOC_THICKNESS::projMatLoc	= glGetUniformLocation(programIds[SHDR_THICKNESS], "P");
	UNILOC_THICKNESS::viewMatLoc	= glGetUniformLocation(programIds[SHDR_THICKNESS], "V");
	UNILOC_THICKNESS::MVPMatLoc		= glGetUniformLocation(programIds[SHDR_THICKNESS], "MVP");
	UNILOC_THICKNESS::viewportLoc	= glGetUniformLocation(programIds[SHDR_THICKNESS], "viewport");
	UNILOC_THICKNESS::spriteSizeLoc = glGetUniformLocation(programIds[SHDR_THICKNESS], "spriteSize");
	UNILOC_THICKNESS::alphaLoc = glGetUniformLocation(programIds[SHDR_THICKNESS], "alpha");

	UNILOC_TEST::projMatLoc = glGetUniformLocation(programIds[SHDR_TEST], "projectionMatrix");
	UNILOC_TEST::invProjMatLoc = glGetUniformLocation(programIds[SHDR_TEST], "invProjectionMatrix");
	UNILOC_TEST::viewportLoc = glGetUniformLocation(programIds[SHDR_TEST], "viewport");
	UNILOC_TEST::renderType = glGetUniformLocation(programIds[SHDR_TEST], "renderType");
	UNILOC_TEST::alphaLoc = glGetUniformLocation(programIds[SHDR_TEST], "alpha");

	UNILOC_SKYBOX::viewMatLoc = glGetUniformLocation(programIds[SHDR_SKYBOX], "V");
	UNILOC_SKYBOX::projMatLoc = glGetUniformLocation(programIds[SHDR_SKYBOX], "P");

	// Variouse textures
	//
	glDeleteTextures(NUM_TEXTURES, textureIds);
	{
		//@ create color texture
		glGenTextures(1, &textureIds[TEX_COLOR]);
		glBindTexture(GL_TEXTURE_2D, textureIds[TEX_COLOR]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowW, windowH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

		//@ create thickness texture
		glGenTextures(1, &textureIds[TEX_THICKNESS]);
		glBindTexture(GL_TEXTURE_2D, textureIds[TEX_THICKNESS]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, windowW, windowH, 0, GL_RED, GL_FLOAT, NULL);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

		//@ create depth texture
		glGenTextures(1, &textureIds[TEX_DEPTH]);
		glBindTexture(GL_TEXTURE_2D, textureIds[TEX_DEPTH]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, windowW, windowH, 0, GL_RED, GL_FLOAT, NULL);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, windowW, windowH, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

		//@ create skybox texture
		glGenTextures(1, &textureIds[TEX_SKYBOX]);
		glBindTexture(GL_TEXTURE_CUBE_MAP, textureIds[TEX_SKYBOX]);
		for (int i = 0; i < 6; ++i){
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, skybox_width, skybox_height, 0, GL_BGR, GL_UNSIGNED_BYTE, (void *)&(texdata_skybox[i].front()));
		}
		glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

		//@ create diffuse wrap texture
		glGenTextures(1, &textureIds[TEX_DIFFUSE_WRAP]);
		glBindTexture(GL_TEXTURE_1D, textureIds[TEX_DIFFUSE_WRAP]);
		glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, &texdata_diffuseWrap.front());
		glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_1D, 0);

		glGenTextures(1, &textureIds[TEX_OUTLINE]);
		glBindTexture(GL_TEXTURE_2D, textureIds[TEX_OUTLINE]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, windowW, windowH, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	glDeleteRenderbuffers(1, &renderbuffer_depth);
	glGenRenderbuffers(1, &renderbuffer_depth);
	glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer_depth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, windowW, windowH);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	//@ create FBO_DEPTH
	glGenFramebuffers(1, &pFBOArray[FBO_DEPTH]);
	glBindFramebuffer(GL_FRAMEBUFFER, pFBOArray[FBO_DEPTH]);
	{
		//@ attach color textures
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureIds[TEX_COLOR], 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureIds[TEX_THICKNESS], 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textureIds[TEX_DEPTH], 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, textureIds[TEX_OUTLINE], 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderbuffer_depth);
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, textureIds[TEX_DEPTH], 0);

		//@ set the list of draw buffers
		GLuint attachments[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
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

	delete timer1;
	timer1 = new TimerGPU();
	delete timer2;
	timer2 = new TimerGPU();

	delete filter;
	filter = BilateralFilter::create(windowW, windowH,
		textureIds[TEX_DEPTH], freq_radius, sigma_s, sigma_r / 255.0f, iterations,
		textureIds[TEX_OUTLINE], dog_radius, dog_sigma, dog_similarity,
		textureIds[TEX_THICKNESS]);
	filter->setMatrix(projectionMatrix, glm::inverse(projectionMatrix));
	
	delete filter_BFGL;
	filter_BFGL = BFGL::create(windowW, windowH, freq_radius, sigma_s, sigma_r, iterations, projectionMatrix, glm::inverse(projectionMatrix));
}

void
render(GLFWwindow* window)
{
	clock_t start = clock();

	if(!pSystem->bPause)	pSystem->simulate(1.0 / FPS);

	clock_t end = clock();
	//cout << "simulation: " << end - start << endl;

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
	glClearDepth(1.0);
	glClear(GL_DEPTH_BUFFER_BIT);
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
		glDrawBuffer(GL_COLOR_ATTACHMENT1);
		glClearColor(1.0, 0.0, 0.0, 0.0);
		glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(programIds[SHDR_DEPTH]);
		{
			glUniform1f(UNILOC_DEPTH::spriteSizeLoc, spriteSize);
			glUniformMatrix4fv(UNILOC_DEPTH::viewMatLoc, 1, GL_FALSE, &viewMatrix[0][0]);
			glUniformMatrix4fv(UNILOC_DEPTH::projMatLoc, 1, GL_FALSE, &projectionMatrix[0][0]);
			//glUniformMatrix4fv(UNILOC_DEPTH::anisotropyMatLoc, 1, GL_FALSE, &pSystem->h_mat[0].x); 

			glEnable(GL_DEPTH_TEST);	// Enable the depth test to obtain the eye-closest surface
			glDrawArrays(GL_POINTS, 0, pSystem->numParticles);
			glDisable(GL_DEPTH_TEST);	// Disable the depth test to accumulate the thickness
		}
		glUseProgram(0);
		checkOpenGL("Depth Shader", __FILE__, __LINE__, false, true);

		// Thickness of the fluid to mimic volume of fluid
		//
		glDrawBuffer(GL_COLOR_ATTACHMENT0);
		glClearColor(0.0, 0.0, 0.0, 0.0);
		glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(programIds[SHDR_THICKNESS]);
		{
			glUniform1f(UNILOC_THICKNESS::spriteSizeLoc, spriteSize);
			glUniform1f(UNILOC_THICKNESS::alphaLoc, fluidAlpha);
			glUniform2fv(UNILOC_THICKNESS::viewportLoc, 1, &viewport[0]);
			glUniform1f(glGetUniformLocation(programIds[SHDR_THICKNESS], "P32"), projectionMatrix[3][2]);
			glUniform1f(glGetUniformLocation(programIds[SHDR_THICKNESS], "P22"), projectionMatrix[2][2]);
			glUniform1f(glGetUniformLocation(programIds[SHDR_THICKNESS], "intensityRange"), zFar - zNear);
			glUniformMatrix4fv(UNILOC_THICKNESS::MVPMatLoc, 1, GL_FALSE, &MVPMatrix[0][0]);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureIds[TEX_DEPTH]);

			glEnable(GL_BLEND);			// Accumlate the thickness using the additive alpha blending
			glBlendFunc(GL_ONE, GL_ONE);
			glDrawArrays(GL_POINTS, 0, pSystem->numParticles);
			glDisable(GL_BLEND);

			glBindTexture(GL_TEXTURE_2D, 0);
		}
		glUseProgram(0);
		checkOpenGL("Thickness Shader", __FILE__, __LINE__, false, true);

		glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glDisableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	//@ orthogonal pjorection from now on
	glm::mat4 invProjectionMatrix	= glm::inverse(projectionMatrix);

	glBindFramebuffer(GL_FRAMEBUFFER, pFBOArray[FBO_DEPTH]);
	timer1->start();
	if (dImageType == DIMG_BILATERAL_FILTERED){
		filter_BFGL->run(textureIds[TEX_DEPTH]);
	}
	else if (dImageType == DIMG_SEPERABLE_FILTERED){
		filter->filter(renderType == RenderTypes::RENDER_CEL || renderType == RenderTypes::RENDER_OUTLINE);
	}
	float elapsedTime = timer1->stop();
	timer2->start();
	//filter->filterThickness();
	if (timer2->isEnabled()){
		std::cout << timer2->stop() << std::endl;
	}

	//@ final scene
	//
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//	float scaledZNear = zNear*((float)windowW/pplaneWidth);
	{
		glUseProgram(programIds[SHDR_SKYBOX]);
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_CUBE_MAP, textureIds[TEX_SKYBOX]);

			glUniformMatrix4fv(UNILOC_SKYBOX::viewMatLoc, 1, GL_FALSE, &viewMatrix[0][0]);
			glUniformMatrix4fv(UNILOC_SKYBOX::projMatLoc, 1, GL_FALSE, &projectionMatrix[0][0]);

			glBindBuffer(GL_ARRAY_BUFFER, finalVbo[0]);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// GL_QUADS is dprecated.

			glDisableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
		glUseProgram(0);
		checkOpenGL("Skybox Shader.", __FILE__, __LINE__, false, true);

		glUseProgram(programIds[SHDR_TEST]);
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureIds[TEX_DEPTH]);

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, textureIds[TEX_THICKNESS]);

			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_1D, textureIds[TEX_DIFFUSE_WRAP]);

			glActiveTexture(GL_TEXTURE3);
			glBindTexture(GL_TEXTURE_2D, textureIds[TEX_OUTLINE]);

			glActiveTexture(GL_TEXTURE4);
			glBindTexture(GL_TEXTURE_CUBE_MAP, textureIds[TEX_SKYBOX]);

			glUniform2fv(UNILOC_TEST::viewportLoc, 1, &viewport[0]);
			glUniform1f(UNILOC_TEST::alphaLoc, fluidAlpha);
			glUniform1i(UNILOC_TEST::renderType, renderType);
			glUniform1i(glGetUniformLocation(programIds[SHDR_TEST], "genNormal"), genNormal);

			glUniformMatrix4fv(UNILOC_TEST::projMatLoc, 1, GL_FALSE, &projectionMatrix[0][0]);
			glUniformMatrix4fv(UNILOC_TEST::invProjMatLoc, 1, GL_FALSE, &invProjectionMatrix[0][0]);
			//glUniformMatrix4fv(UNILOC_TEST::modelViewMatLoc, 1, GL_FALSE, &viewMatrix[0][0]);

			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, finalVbo[0]);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// GL_QUADS is dprecated.
			glDisable(GL_BLEND);

			glDisableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glBindTexture(GL_TEXTURE_1D, 0);
			glBindTexture(GL_TEXTURE_2D, 0);
			glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
		}
		glUseProgram(0);
		checkOpenGL("Final Render Shader", __FILE__, __LINE__, false, true);
	}

	
	end = clock();
	//cout << "total: " << end - start << endl;

	std::stringstream ss;
	ss << "< STATUS >" << std::endl;
	std::string str;
	switch (renderType)
	{
	case RenderTypes::RENDER_TEST:			str = "Test Shading";		break;
	case RenderTypes::RENDER_CEL:			str = "Cel Shading";		break;
	case RenderTypes::RENDER_OUTLINE:		str = "Outline";			break;
	case RenderTypes::RENDER_TEST_R:		str = "Test Shading + Reflection + Refraction";	break;
	case RenderTypes::RENDER_FLAT:			str = "Depth";				break;
	case RenderTypes::RENDER_SHADED:		str = "Shaded";				break;
	case RenderTypes::RENDER_THICKNESS:		str = "Thickness";			break;
	default:								str = "";					break;
	}
	ss << "Rendering mode : " << str << std::endl;
	switch (dImageType)
	{
	case DepthImageTypes::DIMG_BILATERAL_FILTERED:		str = "bilateral filtered using compute shader";		break;
	case DepthImageTypes::DIMG_SEPERABLE_FILTERED:		str = "bilateral filtered using CUDA";		break;
	case DepthImageTypes::DIMG_ORIGINAL:				str = "Original";							break;
	default:											str = "";									break;
	}
	ss << "Depth Image : " << str << std::endl;
	ss << "Kernel Radius : " << freq_radius << std::endl;
	ss << "Sigma R : " << sigma_r << std::endl;
	ss << "Sigma S : " << sigma_s << std::endl;
	ss << "# of Filter Iteration : " << iterations << std::endl;
	if (renderType == RenderTypes::RENDER_CEL || renderType == RenderTypes::RENDER_OUTLINE){
		ss << "DOG Kernel Radius : " << dog_radius << std::endl;
		ss << "DOG Sigma : " << dog_sigma << std::endl;
		ss << "DOG Similarity : " << dog_similarity << std::endl;
	}

	//if (timer1->isEnabled()){
	//	ss << "Elapsed time : " << elapsedTime << std::endl;
	//}

	font->setStyle(FONT_STYLE_BOLD);
	font->setColor(0.0f, 0.0f, 0.0f, 1.0f);
	font->drawText(0, 0, ss.str().c_str());

	font->setStyle(FONT_STYLE_REGULAR);
	font->setColor(1.0f, 1.0f, 1.0f, 1.0f);
	font->drawText(0, 0, ss.str().c_str());
	glUseProgram(0);
}

void
keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (mods == GLFW_MOD_CONTROL){
		if (key == GLFW_KEY_T && action == GLFW_PRESS){
			if (timer1->isEnabled()){
				timer1->report();
				timer1->disable();

				timer2->report();
				timer2->disable();
			}
			else{
				timer1->enable();

				timer2->enable();
			}
		}
		if (glfwGetKey(window, GLFW_KEY_K)){
			freq_radius += 5.0f;
			if (freq_radius > 50.0f){
				freq_radius = 5.0f;
			}
		}
		else if (glfwGetKey(window, GLFW_KEY_R)){
			sigma_r += 1;
			if (sigma_r > 50){
				sigma_r = 0;
			}
		}
		else if (glfwGetKey(window, GLFW_KEY_S)){
			sigma_s += 1;
			if (sigma_s > 50){
				sigma_s = 1;
			}
		}
		else if (glfwGetKey(window, GLFW_KEY_I)){
			++iterations;
			if (iterations > 10){
				iterations = 1;
			}
		}
		filter->setBFParam(freq_radius, spriteSize * (sigma_s / 255.0f), sigma_r * (zFar - zNear) / 255.0f, iterations);
		filter_BFGL->setParameters(freq_radius, spriteSize * (sigma_s / 255.0f), sigma_r * (zFar - zNear) / 255.0f, iterations);
		return;
	}
	else if (mods == GLFW_MOD_ALT){
		if (glfwGetKey(window, GLFW_KEY_K)){
			dog_radius += 1;
			if (dog_radius > 60){
				dog_radius = 1;
			}
		}
		else if (glfwGetKey(window, GLFW_KEY_S)){
			dog_sigma += 1;
			if (dog_sigma > 60){
				dog_sigma = 1;
			}
		}
		else if (glfwGetKey(window, GLFW_KEY_D)){
			dog_similarity -= 0.01f;
			if (dog_similarity < 0.1f){
				dog_similarity = 0.99f;
			}
		}
		filter->setDOGParam(dog_radius, dog_sigma, dog_similarity);
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

	case GLFW_KEY_SLASH:
		if (action == GLFW_PRESS){
			time_t _time = time(NULL);
			tm tm;
			localtime_s(&tm, &_time);
			std::stringstream ss;

			GLfloat *texData = new GLfloat[screenW * screenH];
			for (int i = 0; i < screenW * screenH; ++i){
				texData[i] = 500.0f * abs(screenshot1[i] - screenshot2[i]);
				if (texData[i] > 1.0f){
					texData[i] = 1.0f;
				}
			}

			ss << tm.tm_hour << "_" << tm.tm_min << "_" << tm.tm_sec << ".png";
			saveAsPNG(ss.str().c_str(), texData, screenW, screenH);
			delete[] texData;
		}
		break;
	case GLFW_KEY_COMMA:
		if (action == GLFW_PRESS){
			screenshot1.clear();
			screenshot1.resize(screenW * screenH);

			glBindTexture(GL_TEXTURE_2D, textureIds[TEX_DEPTH]);
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, screenshot1.data());
		}
		break;
	case GLFW_KEY_PERIOD:
		if (action == GLFW_PRESS){
			screenshot2.clear();
			screenshot2.resize(screenW * screenH);

			glBindTexture(GL_TEXTURE_2D, textureIds[TEX_DEPTH]);
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, screenshot2.data());
		}
		break;

	case GLFW_KEY_APOSTROPHE:
		if (action == GLFW_PRESS){
			genNormal = !genNormal;
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