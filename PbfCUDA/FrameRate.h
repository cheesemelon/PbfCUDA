#pragma once

#include <deque>

class FrameRate{
protected:
	std::deque<double> m_deltaTimes;
	double m_sumDeltaTimes;

	double m_MPF;
	int m_numFrames;

	unsigned int m_maxRecordSize;

	FrameRate(unsigned int maxRecordSize)
		: m_sumDeltaTimes(0.0), m_MPF(0.0), m_numFrames(0), m_maxRecordSize(maxRecordSize)
	{ }
public:
	static FrameRate* create(unsigned int maxRecordSize = 30)
	{
		return new FrameRate(maxRecordSize);
	}

	//@ Record and return a deltatime.
	double elapse();

	//@ Averaged FPS.
	double getFPS() const;

	//@ Milliseconds per Frame.
	//@ Good MPF : 16.6666ms for 60fps, 33.3333ms for 30fps.
	double getMPF() const{ return m_MPF; }

	double getDeltaTime() const;
};