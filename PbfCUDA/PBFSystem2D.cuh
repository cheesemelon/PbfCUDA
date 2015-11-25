
#pragma once

//#include <iostream>
//#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
//#include <thrust/device_ptr.h>
//#include <thrust/sort.h>
//#include <math_constants.h>
//#include <device_functions.h>

#include "Common.h"
//#include <math_functions.h>
//#include <time.h>

#define NUM_MAX_PARTICLES			160000
#define NUM_MAX_NEIGHBORS			26

typedef unsigned int	uint;
typedef unsigned short	ushort;

enum MOUSE_TYPE
{
	EMITTER = 0,
	REPULSOR,
	ATTRACTOR,
};

enum predictionType
{
	EXTERNAL,
	FULL,
};

struct SimulationParameters_f4
{
	float4 gravity;
	float4 workspace;
	float4 corner;
};

struct SimulationParameters_f
{
	//float	frameTime;
	float	poly6_precomputed;
	float	spiky_precomputed;
	float	d_rest;
	float	m;
	float	h;
	float	gas_k;
	float	visc_u;
	float	cfm_e;
	float	ap_dq;
	float	ap_k;
	float	ap_n;
	float	xsph_x;
	float	vc_e;
	float	cellSize;

	float	wallFriction;
	float	wallRestitution;
};

struct SimulationParameters_i
{
	//uint	fps;
	uint	nCellX;
	uint	nCellY;
	uint	nCellZ;
	uint	nCells;
};

class PBFSystem2D
{
public:
	PBFSystem2D();
	~PBFSystem2D();

	void	initParticleSystem(SimulationParameters_f4 &pf4);
	void	createParticle(float4 &pos, float4 &vel);
	void	createParticleNbyN(uint n, float spacing, float4 pos, float4 vel);
	void	simulate(float dt);
	void	prediction(float dt);
	void	modifiedPrediction(float dt);
	void	neighborSearch();	
	void	solve(float dt);
	void	update(float dt);
	void	xsph_vorticity();
	void	vorticityConfinement();
	void	collisionResponse();
	void	setSimulationParameters();

	//@ system flags
	bool	bPause;
	bool	bMouseActive;
	bool	bSystemReady;
	bool	bDiagnosis;
	bool	bVCOn;
	bool	bXSPHOn;
	bool	bPlot;

	//@ variables on host memory
	SimulationParameters_f4 hSimParams_f4;
	SimulationParameters_f	hSimParams_f;
	SimulationParameters_i	hSimParams_i;
	uint	numParticles;
	uint	numBlocks;
	uint	numThreads;
	uint	substep;
	uint	numIteration;

	//@ particle variable lists on host memory
	bool	h_bAlive	[NUM_MAX_PARTICLES];				//@ static arrays for debugging											
	float	h_lmd		[NUM_MAX_PARTICLES];
	float4	h_f			[NUM_MAX_PARTICLES];
	float4	h_p_pred	[NUM_MAX_PARTICLES];
	float4	h_dp		[NUM_MAX_PARTICLES];
	float4	h_vx		[NUM_MAX_PARTICLES];
	float	*h_d;											//@ page-locked memories for staging area between host & device
	float4	*h_v;
	float4	*h_p;
	float4	*h_mat;

	//@ hash dump for debugging
	uint	h_numNeighbors	[NUM_MAX_PARTICLES];
	uint	h_cellIndex		[NUM_MAX_PARTICLES];		
	uint	h_particleIndex	[NUM_MAX_PARTICLES];
	uint	h_startCellIndex[NUM_MAX_PARTICLES];
	uint	h_debug			[NUM_MAX_PARTICLES];

	//@ particle variable lists on device(global) memory
	float	*d_lmd;
	float4	*d_f;
	float4	*d_p_pred;
	float4	*d_p_sorted;
	float4	*d_p_sorted_old;
	float4	*d_dp;
	float4	*d_vx;
	float	*d_d;
	float4	*d_v;
	float4	*d_v_sorted;
	float4	*d_p;
	float4	*d_mat;

	uint	*d_numNeighbors;
	uint	*d_neighborIndex;
	uint	*d_cellIndex;
	uint	*d_particleIndex;
	uint	*d_startCellIndex;
	uint	*d_debug;

	//@ temporary variables
	float4	h_mousePos;
	MOUSE_TYPE mouseType;
};

