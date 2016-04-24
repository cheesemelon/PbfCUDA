#pragma once

#include "FrameRate.h"
#include "gl\glfw3.h"

double
FrameRate::elapse()
{
	double currentTime = glfwGetTime();

	// Calc milliseconds per frame
	static double MPF_lastTime = currentTime;
	++m_numFrames;
	if (currentTime - MPF_lastTime >= 1.0){
		m_MPF = 1000.0 / static_cast<double>(m_numFrames);
		m_numFrames = 0;
		MPF_lastTime += 1.0;
	}

	// Calc delta time
	static double dt_lastTime = currentTime;
	double deltaTime = currentTime - dt_lastTime;
	m_sumDeltaTimes += deltaTime;
	m_deltaTimes.push_back(deltaTime);
	if (m_deltaTimes.size() > m_maxRecordSize){
		m_sumDeltaTimes -= m_deltaTimes.front();
		m_deltaTimes.pop_front();
	}
	dt_lastTime = currentTime;

	return deltaTime;
}

double
FrameRate::getFPS() const
{
	if (m_deltaTimes.empty()){
		return 0.0;
	}

	double average = m_sumDeltaTimes / static_cast<double>(m_deltaTimes.size());
	return 1.0 / average;
}

double
FrameRate::getDeltaTime() const
{
	if (m_deltaTimes.empty()){
		return 0.0;
	}

	return m_deltaTimes.back();
}