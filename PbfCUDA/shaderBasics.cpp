
#include "shaderBasics.h"

#include <iostream>

using namespace std;

void
printMat4(glm::mat4 &pMat)
{
	cout << pMat[0].x << ", " << pMat[1].x << ", " << pMat[2].x << ", " << pMat[3].x << endl;
	cout << pMat[0].y << ", " << pMat[1].y << ", " << pMat[2].y << ", " << pMat[3].y << endl;
	cout << pMat[0].z << ", " << pMat[1].z << ", " << pMat[2].z << ", " << pMat[3].z << endl;
	cout << pMat[0].w << ", " << pMat[1].w << ", " << pMat[2].w << ", " << pMat[3].w << endl;
}

void
printVec4(glm::vec4 &pVec)
{
	cout << pVec.x << ", " << pVec.y << ", " << pVec.z << ", " << pVec.w << endl;
}

char* textFileRead(const char *fn)
{
	FILE *fp;
	char *content = NULL;

	int count = 0; 

	if(fn != NULL)
	{
		fp = fopen(fn, "rt");

		if(fp != NULL)
		{
			fseek(fp, 0, SEEK_END);
			count = ftell(fp);
			rewind(fp);

			if(count > 0)
			{
				content = (char*)malloc(sizeof(char)*(count + 1));
				count = fread(content, sizeof(char), count, fp);
				content[count] = '\0';
			}

			fclose(fp);
		}
	}
	return content;
}

void printShaderInfoLog(GLuint obj, const char* shaderName)
{
	int infoLogLength = 0;
	int charsWritten = 0;
	char *infoLog;

	glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infoLogLength);

	if(infoLogLength > 0)
	{
		cout << shaderName << endl;

		infoLog = (char*)malloc(infoLogLength);
		glGetShaderInfoLog(obj, infoLogLength, &charsWritten, infoLog);
		cout << infoLog << endl;
		free(infoLog);
	}
}

void printProgramInfoLog(GLuint obj)
{
	int infoLogLength = 0;
	int charsWritten = 0;
	char *infoLog;

	glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infoLogLength);

	if(infoLogLength > 0)
	{
		infoLog = (char*)malloc(infoLogLength);
		glGetProgramInfoLog(obj, infoLogLength, &charsWritten, infoLog);
		cout << infoLog << endl;
		free(infoLog);
	}
}

GLuint
setShaders(const char* vShaderFileName, const char* fShaderFileName)
{
	GLuint v, f;

	char *vs = NULL; 
	char *fs = NULL;

	v = glCreateShader(GL_VERTEX_SHADER);
	vs = textFileRead(vShaderFileName);

	f = glCreateShader(GL_FRAGMENT_SHADER);
	fs = textFileRead(fShaderFileName);

	const char *ff = fs;
	const char *vv = vs;

	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	free(vs); free(fs);

	glCompileShader(v);
	glCompileShader(f);

	printShaderInfoLog(v, vShaderFileName);
	printShaderInfoLog(f, fShaderFileName);

	GLuint	programID = glCreateProgram();
	glAttachShader(programID, v);
	glAttachShader(programID, f);

	glLinkProgram(programID);

	printProgramInfoLog(programID);

	return	programID;
}

bool
checkOpenGL(const char* message, const char* file, int line, bool exitOnError, bool report)
{
	GLenum	errorCode = glGetError();
	if (errorCode != GL_NO_ERROR)
	{
		if (report)
		{
			cerr << "OpenGL: ";
			if (file)		cerr << file;
			if (line != -1)	cerr << ":" << line;
			if (message)	cerr << " " << message;

			switch (errorCode)
			{
			case GL_INVALID_ENUM:					cout << " invalid enum" << endl;					break;
			case GL_INVALID_VALUE:					cout << " invalid value" << endl;					break;
			case GL_INVALID_OPERATION: 				cout << " invalid operation" << endl;				break;
			case GL_INVALID_FRAMEBUFFER_OPERATION: 	cout << " invalid framebuffer operation" << endl;	break;
			case GL_OUT_OF_MEMORY:				 	cout << " out of memory" << endl; 					break;
			}
		}

		if (exitOnError)	exit(errorCode);

		return	false;
	}

	return	true;
}

void
errorCallback(int error, const char* description)
{
	cerr << "####" << description << endl;
}

GLFWwindow*
initOpenGL(int argc, char* argv[], float screenScale, int& screenW, int& screenH, int& windowW, int& windowH)
{
	glfwSetErrorCallback(errorCallback);

	// Init GLFW
	if(!glfwInit())	exit(EXIT_FAILURE);

	// To enable OpenGL 4.1 in OS X
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	//glfwWindowHint(GLFW_DECORATED, GL_FALSE);

	//glfwWindowHint(GLFW_DEPTH_BITS, 32);
	//glfwWindowHint(GLFW_DEPTH_BITS, 16);

	// Create the window
	GLFWmonitor*	monitor = glfwGetPrimaryMonitor();
	int	monitorW, monitorH;
	glfwGetMonitorPhysicalSize(monitor, &monitorW, &monitorH);
	cerr << "Status: Monitor " << monitorW << "mm x " << monitorH << "mm" << endl;

	const GLFWvidmode*	videoMode = glfwGetVideoMode(monitor);
	//screenW = videoMode->width  * screenScale;	
	//screenH = videoMode->height * screenScale;	
//	screenH = screenW;
	screenW = 1024;
	screenH = 1024;

	GLFWwindow* window;
	window = glfwCreateWindow(screenW, screenH, "GLFW", NULL, NULL);
//	window = glfwCreateWindow(screenW, screenH, "GLFW", glfwGetPrimaryMonitor(), NULL);

	if (!window)
	{
		glfwTerminate();
		cerr << "Failed in glfwCreateWindow()" << endl;
		return	NULL;
	}

	// Context
	glfwMakeContextCurrent(window);

	// Clear the background ASAP
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glFlush();
	glfwSwapBuffers(window);

	// Check the size of the window
	glfwGetWindowSize(window, &screenW, &screenH);
	cerr << "Status: Screeen " << screenW << " x " << screenH << endl;

	glfwGetFramebufferSize(window, &windowW, &windowH);
	cerr << "Status: Framebuffer " << windowW << " x " << windowH << endl;


	// Get the OpenGL version and renderer
	cout << "Status: Renderer " << glGetString(GL_RENDERER) << endl;
	cout << "Status: Ventor " << glGetString(GL_VENDOR) << endl;
	cout << "Status: OpenGL " << glGetString(GL_VERSION) << endl;

	// GLSL version for shader loading
	cout << "Status: GLSL " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

	// Vertical sync ...
	//glfwSwapInterval(0);	// Immediate mode
	//glfwSwapInterval(1);	// Tearing possible

	// GLEW: Supported version can be verified in visit http://glew.sourceforge.net/basic.html
	cerr << "Status: GLEW " << glewGetString(GLEW_VERSION) << endl;

	// (1) to circumvent glewInit() error. However, nothing changed.
	// (2) to use glGenVertexArrays
	glewExperimental = true;

	GLenum error = glewInit();
	checkOpenGL("glewInit()", __FILE__, __LINE__, false, false);
	if (error != GLEW_OK)
	{
		cerr << "ERROR: " << glewGetErrorString(error) << endl;
		return	0;
	}
	return	window;
}