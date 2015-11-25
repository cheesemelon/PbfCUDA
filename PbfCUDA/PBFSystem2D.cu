
// CUDA
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <math_constants.h>
#include <device_functions.h>
#include <math_functions.h>
#include <device_launch_parameters.h>

#include <time.h>

#include "PBFSystem2D.cuh"
#include "Common.h"


using namespace std;


#define NUM_MAX_PARTICLES			160000
#define NUM_MAX_NEIGHBORS			26

//@ texture references
texture<float4> tex_pos;
texture<float4> tex_vel;
texture<float4> tex_sortedPos;
texture<float4> tex_OldSortedPos;
texture<float4> tex_sortedVel;
texture<float4>	tex_vorticity;
texture<float>	tex_density;
texture<float>	tex_lmd;
texture<uint>	tex_particleIndex;
texture<uint>	tex_cellIndex;
texture<uint>	tex_startCellIndex;
texture<uint>	tex_neighborIndex;
texture<uint>	tex_numNeighbors;

__device__ __constant__ SimulationParameters_f4 cSimParams_f4;
__device__ __constant__ SimulationParameters_f cSimParams_f;
__device__ __constant__ SimulationParameters_i cSimParams_i;

inline __device__ void mat3Add(float3 *mat1, float3 *mat2, float3 *result)
{
	result[0] = mat1[0] + mat2[0];
	result[1] = mat1[1] + mat2[1];
	result[2] = mat1[2] + mat2[2];
}	

inline __device__ void mat3Sub(float3 *mat1, float3 *mat2, float3 *result)
{
	result[0] = mat1[0] - mat2[0];
	result[1] = mat1[1] - mat2[1];
	result[2] = mat1[2] - mat2[2];
}

inline __device__ void mat3Mult(float3 *mat1, float3 *mat2, float3 *result)
{
	float3 temp[3];

	temp[0].x = mat1[0].x*mat2[0].x + mat1[0].y*mat2[1].x + mat1[0].z*mat2[2].x;
	temp[0].y = mat1[0].x*mat2[0].y + mat1[0].y*mat2[1].y + mat1[0].z*mat2[2].y;
	temp[0].z = mat1[0].x*mat2[0].z + mat1[0].y*mat2[1].z + mat1[0].z*mat2[2].z;

	temp[1].x = mat1[1].x*mat2[0].x + mat1[1].y*mat2[1].x + mat1[1].z*mat2[2].x;
	temp[1].y = mat1[1].x*mat2[0].y + mat1[1].y*mat2[1].y + mat1[1].z*mat2[2].y;
	temp[1].z = mat1[1].x*mat2[0].z + mat1[1].y*mat2[1].z + mat1[1].z*mat2[2].z;

	temp[2].x = mat1[2].x*mat2[0].x + mat1[2].y*mat2[1].x + mat1[2].z*mat2[2].x;
	temp[2].y = mat1[2].x*mat2[0].y + mat1[2].y*mat2[1].y + mat1[2].z*mat2[2].y;
	temp[2].z = mat1[2].x*mat2[0].z + mat1[2].y*mat2[1].z + mat1[2].z*mat2[2].z;

	result[0] = temp[0];
	result[1] = temp[1];
	result[2] = temp[2];
}

inline __device__ void mat3Mult(float f, float3 *mat1, float3 *result)
{
	result[0] = mat1[0]*f;
	result[1] = mat1[1]*f;
	result[2] = mat1[2]*f;
}

inline __device__ void mat3Div(float f, float3 *mat1, float3 *result)
{
	if(f > 0.0f)
	{
		result[0] = mat1[0]/f;
		result[1] = mat1[1]/f;
		result[2] = mat1[2]/f;
	}
}

inline __device__ void mat3Transpose(float3 *mat, float3 *result)
{
	float3 temp[3];

	temp[0].x = mat[0].x;
	temp[0].y = mat[1].x;
	temp[0].z = mat[2].x;

	temp[1].x = mat[0].y;
	temp[1].y = mat[1].y;
	temp[1].z = mat[2].y;

	temp[2].x = mat[0].z;
	temp[2].y = mat[1].z;
	temp[2].z = mat[2].z;

	result[0] = temp[0];
	result[1] = temp[1];
	result[2] = temp[2];
}

inline __device__ void mat3Transpose(float3 *mat)
{
	float temp = 0;

	temp = mat[0].y;
	mat[0].y = mat[1].x;
	mat[1].x = temp;

	temp = mat[0].z;
	mat[0].z = mat[2].x;
	mat[2].x = temp;

	temp = mat[1].z;
	mat[1].z = mat[2].y;
	mat[2].y = temp;
}

inline __device__ float mat3Trace(float3 *mat)
{
	return mat[0].x + mat[1].y + mat[2].z;
}

inline __device__ float mat3Det(float3 *mat)
{
	return 
			mat[0].x*(mat[1].y*mat[2].z - mat[1].z*mat[2].y) 
		-	mat[0].y*(mat[1].x*mat[2].z - mat[1].z*mat[2].x) 
		+	mat[0].z*(mat[1].x*mat[2].y - mat[1].y*mat[2].x);
}

inline __device__ float3 mat3CofactorRow1(float3 *mat)
{
	return make_float3(mat[1].y*mat[2].z - mat[1].z*mat[2].y, mat[1].z*mat[2].x - mat[1].x*mat[2].z, mat[1].z*mat[2].y - mat[1].y*mat[2].x);
}

inline __device__ float W_poly6(float r)
{
	float h2_minus_r2 = cSimParams_f.h*cSimParams_f.h - r*r;
	return cSimParams_f.poly6_precomputed*h2_minus_r2*h2_minus_r2*h2_minus_r2;
}

inline __device__ float4 grad_W_spiky(float4 r, float r_l)
{
	float h_minus_rl = cSimParams_f.h - r_l;
	return cSimParams_f.spiky_precomputed*h_minus_rl*h_minus_rl*normalize( r );
}

inline __device__ float artificialPressure( float r_l )
{
	float	h_minus_r	= cSimParams_f.h - r_l;
	float	h_minus_dq	= cSimParams_f.h - cSimParams_f.ap_dq;
	float	s			= (h_minus_r*h_minus_r*h_minus_r)/(h_minus_dq*h_minus_dq*h_minus_dq);

	float	pow_s = 1;
	uint	n = cSimParams_f.ap_n;
	for( uint i = 0; i < n; i++ )
		pow_s *= s;

	return -cSimParams_f.ap_k*pow_s;
}

inline __device__ uint calcCellIndex(float4 predictedPos)
{
	float4 cellPos = (predictedPos - cSimParams_f4.corner)/cSimParams_f.cellSize;

	if( cellPos.x < 0 ) cellPos.x = 0;
	if( cellPos.y < 0 ) cellPos.y = 0;
	if( cellPos.z < 0 ) cellPos.z = 0;

	uint i = (uint)cellPos.x;
	uint j = (uint)cellPos.y;
	uint k = (uint)cellPos.z;

	if( i >= cSimParams_i.nCellX ) i = cSimParams_i.nCellX - 1;
	if( j >= cSimParams_i.nCellY ) j = cSimParams_i.nCellY - 1;
	if( k >= cSimParams_i.nCellZ ) j = cSimParams_i.nCellZ - 1;

	return i + k*cSimParams_i.nCellX + j*cSimParams_i.nCellX*cSimParams_i.nCellZ;
}

inline __device__ uint calcCellIndex(int i, int j, int k)
{
	if( i < 0 ) i = 0;
	if( j < 0 ) j = 0;
	if( k < 0 ) k = 0;
	if( i >= cSimParams_i.nCellX ) i = cSimParams_i.nCellX - 1;
	if( j >= cSimParams_i.nCellY ) j = cSimParams_i.nCellY - 1;
	if( k >= cSimParams_i.nCellZ ) j = cSimParams_i.nCellZ - 1;

	uint _i = i;
	uint _j = j;
	uint _k = k;

	return _i + _k*cSimParams_i.nCellX + _j*cSimParams_i.nCellX*cSimParams_i.nCellZ;
}

__global__ void g_createParticle(
	//@ input
	uint index, 
	float4 particleInitVel,
	float4 particleInitPos, 
	
	//@ output
	float4 *pVel,
	float4 *pPos)
{
	if( index >= NUM_MAX_PARTICLES ) return;

	pPos[index]	= particleInitPos;
	pVel[index]	= particleInitVel;
}

__global__ void g_createParticleNbyN(
	//@ input
	uint numParticle,
	uint n, uint k,
	float spacing,
	float4 particleInitVel,
	float4 particleInitPos,

	//@ output
	float4 *pVel,
	float4 *pPos)
{
	for(uint j = 0; j < k; j++)
	for(uint i = 0; i < n; i++)
	{
		float4 dsp = make_float4( i*spacing*cSimParams_f.h, 0, j*spacing*cSimParams_f.h, 0);
		pPos[numParticle + i + j*n] = particleInitPos + dsp;
		pVel[numParticle + i + j*n] = particleInitVel;
	}
}

__global__ void setDeviceUintArray(
	//@ input
	uint n,
	bool bIncremental,
	uint data,
	
	//@ output
	uint *dArray)
{
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if( index >= n ) return;

	//@ output
	if( bIncremental )
		dArray[index] = index;
	else
		dArray[index] = data;
}

__global__ void g_prediction(
	//@ input
	uint numParticles,
	float4 *pPos, 
	float *pDensity,
	float dt,
	
	float4 *pForce,
	float4 *pVel,
	float4 *pPredPos,
	uint *pCellIndex)
{
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if( index >= numParticles ) return;

	float4	a;
	float4	v	= tex1Dfetch(tex_vel, index);
	float4	p	= tex1Dfetch(tex_pos, index);

	//@ prediction 
	a = pForce[index]/tex1Dfetch(tex_density, index) + cSimParams_f4.gravity;
	v = v + a*dt;
	float4	p_pred		= p + v*dt;

	//@ output
	pForce[index]		= make_float4(0, 0, 0, 0);
	pPredPos[index]		= p_pred;
	pCellIndex[index]	= calcCellIndex(p_pred); 
}

__global__ void g_findStartCellIndex(
	//@ input
	uint numParticles,
	uint *pCellIndex,
	uint *pParticleIndex,
	float4 *pPredPos,

	//@ output
	uint *pStartIndex,
	float4 *pSortedPos,
	float4 *pOldSortedPos)
{
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if(index >= numParticles ) return;

	uint cellIndex = pCellIndex[index];
	uint nextCellIndex = cellIndex;
	if( index < numParticles - 1 )
		nextCellIndex = pCellIndex[index + 1];

	if(index == 0 )
	{
		pStartIndex[cellIndex] = index;

		if(cellIndex != nextCellIndex)
			pStartIndex[nextCellIndex] = index + 1;
	}
	else if(cellIndex != nextCellIndex)
	{
		pStartIndex[nextCellIndex] = index + 1;
	}

	uint pIndex = pParticleIndex[index];
	pSortedPos[index] = pPredPos[pIndex];
}

__global__ void g_setNeighborList(
	//@ input
	uint numParticles,
	uint n,
	float4 *pSortedPos,
	uint *pCellIndex,
	uint *pStartCellIndex,
	uint *pParticleIndex,

	//@ output
	uint *pNumNeighbors,
	uint *pNeighborIndex)
{
	uint bigIndex = blockDim.x*blockIdx.x + threadIdx.x;

	for( uint work = 0; work < n; work++ )
	{
		uint index = bigIndex*n + work;
		if( index >= numParticles ) return;

		float4 p_pred_i = tex1Dfetch(tex_sortedPos, index);
	
		int i = (p_pred_i.x - cSimParams_f4.corner.x)/cSimParams_f.cellSize;
		int j = (p_pred_i.y - cSimParams_f4.corner.y)/cSimParams_f.cellSize;
		int k = (p_pred_i.z - cSimParams_f4.corner.z)/cSimParams_f.cellSize;
	
		uint neighborCount = 0;
		for( int _j = -1; _j <= 1; _j++ )
		for( int _k = -1; _k <= 1; _k++ )
		for( int _i = -1; _i <= 1; _i++ )
		{
			if(neighborCount >= NUM_MAX_NEIGHBORS) break;

			uint neighborCellIndex = calcCellIndex(i + _i, j + _j, k + _k);
			uint startCellIndex = pStartCellIndex[neighborCellIndex];

		//	uint neighborPerCell = 0;
			while(tex1Dfetch(tex_cellIndex, startCellIndex) == neighborCellIndex && neighborCount < NUM_MAX_NEIGHBORS/*&& neighborPerCell < NUM_MAX_NEIGHBORS_PER_CELL*/ )
			//while(__ldg(&pCellIndex[startCellIndex]) == neighborCellIndex && neighborCount < NUM_MAX_NEIGHBORS/*&& neighborPerCell < NUM_MAX_NEIGHBORS_PER_CELL*/ )
			{
				if( startCellIndex != index) 
				{
					float4	p_pred_j	= tex1Dfetch(tex_sortedPos, startCellIndex);

					float4	r			= p_pred_i - p_pred_j;
					float	r_len		= length(r);

					if(r_len < cSimParams_f.h)
					{
						pNeighborIndex[index*NUM_MAX_NEIGHBORS + neighborCount] = startCellIndex;
						neighborCount++;
					}
				}
				startCellIndex++;
			}
		}

		pNumNeighbors[index] = neighborCount;
	}
}

__global__ void g_computeLambda(
	//@ input
	uint numParticles,
	float4 *pSortedPos,
	uint *pNumNeighbors,
	uint *pNeighborIndex,

	//@ output
	float *pDensity,
	float *pLambda)
{
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if(index >= numParticles ) return;

	float4 p_pred_i = tex1Dfetch(tex_sortedPos, index);

	float	d_i = cSimParams_f.m*W_poly6(0);
	float4	grad_i_c = make_float4(0, 0, 0, 0);
	float4	grad_j_c = make_float4(0, 0, 0, 0);
	float	sum = 0;

	uint numNeighbors = pNumNeighbors[index];
	for( uint i = 0; i < NUM_MAX_NEIGHBORS; i++ )
	{
		if( i == numNeighbors ) break;

		//@ Index of neighbor in sorted particle index list
		uint neighborIndex = tex1Dfetch(tex_neighborIndex, index*NUM_MAX_NEIGHBORS + i);

		//@ referencing from sorted list
		float4 p_pred_j		= tex1Dfetch(tex_sortedPos, neighborIndex);
		float4	r			= p_pred_i - p_pred_j;
		float	r_len		= length(r);

		if( r_len < cSimParams_f.h )
		{
			//@ density interpolation
			d_i += cSimParams_f.m*W_poly6(r_len);

			//@ gradient of constraints
			float4 grad = (cSimParams_f.m/cSimParams_f.d_rest)*grad_W_spiky(r, r_len);
			grad_i_c += grad;
			grad_j_c = -grad;
			sum += dot(grad_j_c, grad_j_c);			
		}
	}

	sum += dot(grad_i_c, grad_i_c);
	float c = d_i/cSimParams_f.d_rest - 1.0;

	//@ output
	pDensity[index] = d_i;
	if( sum != 0 ) 
		pLambda[index] = -c/(sum + cSimParams_f.cfm_e);
	else
		pLambda[index] = 0;
}

__global__ void g_computePositionDelta(
	//@ input
	uint numParticles,
	float4 *pSortedPos,
	float *pLambda,
	uint *pNumNeighbors,
	uint *pNeighborIndex,

	//@ output
	float4 *pPosDelta)
{
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if(index >= numParticles ) return;

	float4 p_pred_i = tex1Dfetch(tex_sortedPos, index);
	float4	dp		= make_float4(0, 0, 0, 0);
	float	lambda	= pLambda[index];

	uint numNeighbors = pNumNeighbors[index];
	for( uint i = 0; i < NUM_MAX_NEIGHBORS; i++ )
	{
		if( i == numNeighbors ) break;

		uint neighborIndex = tex1Dfetch(tex_neighborIndex, index*NUM_MAX_NEIGHBORS + i);

		//@ referencing from sorted list
		float4 p_pred_j		= tex1Dfetch(tex_sortedPos, neighborIndex);
		float4	r			= p_pred_i - p_pred_j;
		float	r_len		= length(r);

		if( r_len < cSimParams_f.h )
		{
			//@ artificial pressure
			float s = artificialPressure(r_len);

			//@ position delta
			float4 grad = (lambda + pLambda[neighborIndex] + s)*(cSimParams_f.m/cSimParams_f.d_rest)*grad_W_spiky(r, r_len);	
			dp += grad;
		}
	}

	//@ output
	pPosDelta[index] = dp;
}

__global__ void g_correction(
	//@ input
	uint numParticles,
	uint *pParticleIndex,

	//@ output
	float4 *pPredPos,
	float4 *pSortedPos,
	float4 *pDeltaPos)
{
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if( index >= numParticles ) return;

	uint pIndex = pParticleIndex[index];

	float4 temp = pSortedPos[index] + pDeltaPos[index];
	pSortedPos[index] =	temp;
	pPredPos[pIndex] = temp;
	pDeltaPos[index] = make_float4(0, 0, 0, 0);
}

__global__ void g_update(
	//@ input
	uint numParticles,
	float4 *pPredPos,
	float4 *pOldOsortedPos,
	uint *pParticleIndex,
	float dt,

	//@ output
	float4 *pVel,
	float4 *pSortedVel,
	float4 *pPos)
{
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if( index >= numParticles ) return;

	uint pIndex = pParticleIndex[index];

	float4 p_pred = tex1Dfetch(tex_sortedPos, index);
	float4 newVel = (p_pred - pPos[pIndex])/dt;

	//@ output
	pVel[pIndex] = newVel;
	pSortedVel[index] = newVel;
	pPos[pIndex] = p_pred;
}

__global__ void g_anisotrophy(
	//@ input
	uint numParticles,
	uint *pNumNeighbors,
	uint *pParticleIndex,

	//@ output
	float4 *pAnisotropyMatrix
	)
 {
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if( index >= numParticles ) return;

//	uint pIndex = pParticleIndex[index];

	float4 p_pos_i = tex1Dfetch(tex_sortedPos, index);

	float weightSum = 0.0f;
	float4 weightedPos_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	float3 C[3];
	C[0] = make_float3(0.0f, 0.0f ,0.0f );
	C[1] = make_float3(0.0f, 0.0f ,0.0f );
	C[2] = make_float3(0.0f, 0.0f ,0.0f );

	//C[0] = make_float3(1.0f, 0.0f ,0.0f );
	//C[1] = make_float3(0.0f, 1.0f ,0.0f );
	//C[2] = make_float3(0.0f, 0.0f ,1.0f );
	//mat3Mult(0.3, C, C);

	uint numNeighbors = pNumNeighbors[index];
	if( numNeighbors >= 6 )
	{
		for( uint i = 0; i < NUM_MAX_NEIGHBORS; i++ )
		{
			if( i == numNeighbors ) break;

			uint neighborIndex = tex1Dfetch(tex_neighborIndex, index*NUM_MAX_NEIGHBORS + i);

			float4 p_pos_j	= tex1Dfetch(tex_sortedPos, neighborIndex);

			float4	r		= p_pos_i - p_pos_j;
			float	r_len	= length(r);

			if( r_len < cSimParams_f.h )
			{
				float rDivh = r_len/cSimParams_f.h;
				float w = 1.0f - rDivh*rDivh*rDivh;
				weightSum += w;
				weightedPos_i += w*p_pos_j;
			}
		}
		weightedPos_i /= weightSum;

		for( uint i = 0; i < NUM_MAX_NEIGHBORS; i++ )
		{
			if( i == numNeighbors ) break;

			uint neighborIndex = tex1Dfetch(tex_neighborIndex, index*NUM_MAX_NEIGHBORS + i);

			float4 p_pos_j	= tex1Dfetch(tex_sortedPos, neighborIndex);

			float4	r		= p_pos_i - p_pos_j;
			float	r_len	= length(r);

			if( r_len < cSimParams_f.h )
			{
				float rDivh = r_len/cSimParams_f.h;
				float w = 1.0f - rDivh*rDivh*rDivh;

				float4 x = p_pos_j - weightedPos_i;

				float3 XXT[3];
				XXT[0] = make_float3(x.x*x.x, x.x*x.y, x.x*x.z);
				XXT[1] = make_float3(x.x*x.y, x.y*x.y, x.y*x.z);
				XXT[2] = make_float3(x.x*x.z, x.y*x.z, x.z*x.z);

				mat3Mult(w, XXT, XXT);
				mat3Add(C, XXT, C);
			}
		}
		mat3Div(weightSum, C, C);	
	}
	else
	{
		C[0] = make_float3(1.0f, 0.0f ,0.0f );
		C[1] = make_float3(0.0f, 1.0f ,0.0f );
		C[2] = make_float3(0.0f, 0.0f ,1.0f );
	}


	//@ analytic eigenvalue algoritm 
	float eig1, eig2, eig3;

	//// A is always symmetric 
	//float3 A[3];
	//float3 C_T[3];
	//mat3Transpose(C, C_T);
	//mat3Mult(C_T, C, A);

	float p1 = C[0].y*C[0].y + C[0].z*C[0].z + C[1].z*C[1].z;
	if( p1 == 0 ) // C is diagonal
	{
		float3 sort = make_float3(C[0].x, C[1].y, C[2].z);
		float temp = 0;

		if(sort.x > sort.y)
		{
			temp = sort.x;
			sort.x = sort.y;
			sort.y = temp;
		}

		if(sort.y > sort.z)
		{
			temp = sort.y;
			sort.y = sort.z;
			sort.z = temp;
		}

		if(sort.x > sort.y)
		{
			temp = sort.x;
			sort.x = sort.y;
			sort.y = temp;
		}

		eig1 = sort.z;
		eig2 = sort.y; 
		eig3 = sort.x;
	}
	else
	{
		float q = mat3Trace(C)/3.0f;
		float p2 = (C[0].x - q)*(C[0].x - q) + (C[1].y - q)*(C[1].y - q) + (C[2].z - q)*(C[2].z - q) + 2.0f*p1;
		float p = sqrtf(p2/6.0f);
		
		float3 B[3];
		float3 qI[3];
		qI[0] = make_float3(q, 0, 0);
		qI[1] = make_float3(0, q, 0);
		qI[2] = make_float3(0, 0, q);

		mat3Sub(C, qI, B);
		mat3Mult(1.0f/p, B, B);
		float r = mat3Det(B)/2.0f;
		float phi = 0;

		if(r <= -1)
			phi = CUDART_PI_F/3.0f;
		else if(r >= 1)
			phi = 0;
		else
			phi = acosf(r)/3.0f;

		eig1 = q + 2.0f*p*cos(phi);
		eig3 = q + 2.0f*p*cos(phi + (2.0f*CUDART_PI_F/3.0f));
		eig2 = 3.0f*q - eig1 - eig3;
	}

	float3 eigenvector[3];
	float eps = 0.001f; // 10^-5
//	float multiplicity = 2;

	// multiplicity 3
	if(eig1 - eig3 < eps*eig1) 
	{
		eigenvector[0] = make_float3(1, 0, 0);
		eigenvector[1] = make_float3(0, 1, 0);
		eigenvector[2] = make_float3(0, 0, 1);

	//	multiplicity = 3;
	}

	//@ multiplicity 2	
	// eig1 != eig2 & eig2 == eig3
	else if(eig1 - eig2 >= eps*eig1 && eig2 - eig3 <= eps*eig2)
	{
		float3 eigI[3];
		float3 B[3];

		//eigenvector1
		eigI[0] = make_float3(eig1, 0, 0);
		eigI[1] = make_float3(0, eig1, 0);
		eigI[2] = make_float3(0, 0, eig1);
		mat3Sub(C, eigI, B);
		eigenvector[0] = mat3CofactorRow1(B);
		eigenvector[0] = normalize(eigenvector[0]);

		//eigenvector2
		eigI[0] = make_float3(eig2, 0, 0);
		eigI[1] = make_float3(0, eig2, 0);
		eigI[2] = make_float3(0, 0, eig2);
		mat3Sub(C, eigI, B);
		eigenvector[1] = mat3CofactorRow1(B);
		eigenvector[1] = eigenvector[1] - dot(eigenvector[0], eigenvector[1])*eigenvector[0];
		eigenvector[1] = normalize(eigenvector[1]);

		//eigenvector3
		eigenvector[2] = cross(eigenvector[0], eigenvector[1]);
		eigenvector[2] = normalize(eigenvector[2]);

	//	multiplicity = 5;
	}					
	// eig2 != eig3 & eig1 == eig2
	else if(eig2 - eig3 >= eps*eig2 && eig1 - eig2 <= eps*eig1)
	{
		float3 eigI[3];
		float3 B[3];

		//eigenvector1
		eigI[0] = make_float3(eig1, 0, 0);
		eigI[1] = make_float3(0, eig1, 0);
		eigI[2] = make_float3(0, 0, eig1);
		mat3Sub(C, eigI, B);
		eigenvector[0] = mat3CofactorRow1(B);
		eigenvector[0] = normalize(eigenvector[0]);

		//eigenvector2
		eigI[0] = make_float3(eig2, 0, 0);
		eigI[1] = make_float3(0, eig2, 0);
		eigI[2] = make_float3(0, 0, eig2);
		mat3Sub(C, eigI, B);
		eigenvector[1] = mat3CofactorRow1(B);
		eigenvector[1] = eigenvector[1] - dot(eigenvector[0], eigenvector[1])*eigenvector[0];
		eigenvector[1] = normalize(eigenvector[1]);

		//eigenvector3
		eigenvector[2] = cross(eigenvector[0], eigenvector[1]);
		eigenvector[2] = normalize(eigenvector[2]);

	//	multiplicity = 6;
	}

	// multiplicity 1
	else
	{
		float3 eigI[3];
		float3 B[3];

		//eigenvector1
		eigI[0] = make_float3(eig1, 0, 0);
		eigI[1] = make_float3(0, eig1, 0);
		eigI[2] = make_float3(0, 0, eig1);
		mat3Sub(C, eigI, B);
		eigenvector[0] = mat3CofactorRow1(B);
		eigenvector[0] = normalize(eigenvector[0]);

		//eigenvector3
		eigI[0] = make_float3(eig3, 0, 0);
		eigI[1] = make_float3(0, eig3, 0);
		eigI[2] = make_float3(0, 0, eig3);
		mat3Sub(C, eigI, B);
		eigenvector[2] = mat3CofactorRow1(B);
		eigenvector[2] = eigenvector[2] - dot(eigenvector[0], eigenvector[2])*eigenvector[0];
		eigenvector[2] = normalize(eigenvector[2]);

		//eigenvector2
		eigenvector[1] = cross(eigenvector[2], eigenvector[0]);
		eigenvector[1] = normalize(eigenvector[1]);

	//	multiplicity = 1;
	}

	//eigenvector[0] = normalize(eigenvector[0]);
	//eigenvector[1] = normalize(eigenvector[1]);
	//eigenvector[2] = normalize(eigenvector[2]);

	float k_r = 4.0f;
//	float k_s = 1400.0f;
//	float k_n = 0.5f;

	float sigma = eig1/k_r;
	eig2 = eig1 > k_r*eig2 ? sigma : eig2;
	eig3 = eig1 > k_r*eig3 ? sigma : eig3;

	float n = cbrtf(1.0f/(eig1*eig2*eig3));
	eig1 *= n;
	eig2 *= n;
	eig3 *= n;

	//pAnisotropyMatrix[3*index + 0] = make_float4(eig1, eig2, eig3, multiplicity);
	//pAnisotropyMatrix[3*index + 1] = make_float4(eigenvector[0].x, eigenvector[0].y, eigenvector[0].z, dot(eigenvector[0], eigenvector[1]) + dot(eigenvector[1], eigenvector[2]) + dot(eigenvector[0], eigenvector[2]));
	//pAnisotropyMatrix[3*index + 2] = make_float4(eigenvector[1].x, eigenvector[1].y, eigenvector[1].z, n);

	float3 T[3];
	T[0] = make_float3(1, 0, 0);
	T[1] = make_float3(0, 1, 0);
	T[2] = make_float3(0, 0, 1);

	//mat3Transpose(eigenvector);
	//mat3Mult(eigenvector, T, T);

	pAnisotropyMatrix[3*index + 0] = make_float4(T[0]);
	pAnisotropyMatrix[3*index + 1] = make_float4(T[1]);
	pAnisotropyMatrix[3*index + 2] = make_float4(T[2]);
}

__global__ void g_xsph_vorticity(
	//@ input
	uint numParticles,
	//float4 *pSortedVel,
	float4 *pSortedPos,
	float *pDensity,
	uint *pNumNeighbors,
	uint *pParticleIndex,

	//@ output
	float4 *pVel,
	float4 *pVorticity)
{
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if( index >= numParticles ) return;

	uint pIndex = pParticleIndex[index];

	float4 p_vel_i = tex1Dfetch(tex_sortedVel, index); 
	float4 p_pos_i = tex1Dfetch(tex_sortedPos, index);
	float4 new_vel = make_float4(0, 0, 0, 0);
	float4 vorticity = make_float4(0, 0, 0, 0);
	
	uint numNeighbors = pNumNeighbors[index];
	for( uint i = 0; i < NUM_MAX_NEIGHBORS; i++ )
	{
		if( i == numNeighbors ) break;

		uint neighborIndex = tex1Dfetch(tex_neighborIndex, index*NUM_MAX_NEIGHBORS + i);

		//@ referencing from sorted list
		float4 p_vel_j	= tex1Dfetch(tex_sortedVel, neighborIndex); 
		float4 p_pos_j	= tex1Dfetch(tex_sortedPos, neighborIndex);
		float4 v_ij		= p_vel_j - p_vel_i; 

		float4	r		= p_pos_i - p_pos_j;
		float	r_len	= length(r);

		if( r_len < cSimParams_f.h )
		{
			float density_j = pDensity[neighborIndex];
			new_vel += cSimParams_f.xsph_x*(cSimParams_f.m/density_j)*v_ij*W_poly6(r_len);
			vorticity += -(cSimParams_f.m/density_j)*cross(v_ij, grad_W_spiky(r, r_len));
		}
	}

	//output
	pVel[pIndex] = p_vel_i + new_vel;
	pVorticity[index] = vorticity;
}

__global__ void g_vorticityConfinement(
	//@ input
	uint numParticles,
	float4 *pSortedPos,
	float *pDensity,
	float4 *pVorticity,
	uint *pNumNeighbors,
	uint *pParticleIndex,

	//@ output
	float4 *pForce)
{
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if( index >= numParticles ) return;

	uint pIndex = pParticleIndex[index];

	float4 p_pos_i = tex1Dfetch(tex_sortedPos, index);
	float4 location = make_float4(0, 0, 0, 0);
	
	uint numNeighbors = pNumNeighbors[index];
	for( uint i = 0; i < NUM_MAX_NEIGHBORS; i++ )
	{
		if( i == numNeighbors ) break;

		uint neighborIndex = tex1Dfetch(tex_neighborIndex, index*NUM_MAX_NEIGHBORS + i);

		float4 p_pos_j	= tex1Dfetch(tex_sortedPos, neighborIndex);
		float4	r		= p_pos_i - p_pos_j;
		float	r_len	= length(r);

		if( r_len < cSimParams_f.h )
		{
			location += (cSimParams_f.m/pDensity[neighborIndex])*length(pVorticity[neighborIndex])*grad_W_spiky(r, r_len);
		}
	}
	location = normalize(location);

	float4 f_conf = cSimParams_f.vc_e*cross(location, pVorticity[index]);
	pForce[pIndex] = f_conf;
}

__global__ void g_wallCollision(
	//@ input
	uint numParticles,
	float4 *pPos,
	float4 *pVel

	//@ output
	)
{
	uint index = blockDim.x*blockIdx.x + threadIdx.x;
	if( index >= numParticles ) return;

	float4 halfSpace = 0.5*cSimParams_f4.workspace;
	float margin = cSimParams_f.cellSize + cSimParams_f.h;

	float lFriction = 0.01f;
	float rFriction = 0.01f;
	float bFriction = 0.02f;
	float tFriction = 0.01f;
	float frtFriction = 0.01f;
	float bckFriction = 0.01f;
	float restitution = 0.3f;

	float4 x = tex1Dfetch(tex_pos, index);
	float4 v = tex1Dfetch(tex_vel, index);

	//@ left wall
	{
		float4 n = make_float4( 1, 0, 0, 0 );
		float4 v_t;
		float4 v_n;

		if( x.x <= -halfSpace.x + margin )
		{
			x = make_float4( -halfSpace.x + margin, x.y, x.z, 0 );

			float vDotn = dot( v, n );		
			if( vDotn <= 0 )
			{
				v_n = abs(vDotn)*n;
				v_t = v + v_n;

				v = restitution*v_n + (1.0f - lFriction)*v_t;
			}
		}
	}

	//@ right wall
	{
		float4 n = make_float4( -1, 0, 0, 0 );
		float4 v_t;
		float4 v_n;

		if( x.x >= halfSpace.x - margin )
		{
			x = make_float4( halfSpace.x - margin, x.y, x.z, 0 );

			float vDotn = dot( v, n );
			if( vDotn <= 0 )
			{
				v_n = abs(vDotn)*n;
				v_t = v + v_n;

				v = restitution*v_n + (1.0f - rFriction)*v_t;
			}
		}
	}

	//@ back wall
	{
		float4 n = make_float4( 0, 0, 1, 0 );
		float4 v_t;
		float4 v_n;

		if( x.z <= -halfSpace.z + margin )
		{
			x = make_float4( x.x, x.y, -halfSpace.z + margin, 0 );

			float vDotn = dot( v, n );		
			if( vDotn <= 0 )
			{
				v_n = abs(vDotn)*n;
				v_t = v + v_n;

				v = restitution*v_n + (1.0f - bckFriction)*v_t;
			}
		}
	}

	//@ front wall
	{
		float4 n = make_float4( 0, 0, -1, 0 );
		float4 v_t;
		float4 v_n;

		if( x.z >= halfSpace.z - margin )
		{
			x = make_float4( x.x, x.y, halfSpace.z - margin, 0 );

			float vDotn = dot( v, n );		
			if( vDotn <= 0 )
			{
				v_n = abs(vDotn)*n;
				v_t = v + v_n;

				v = restitution*v_n + (1.0f - frtFriction)*v_t;
			}
		}
	}


	//@ botton wall
	{
		float4 n = make_float4( 0, 1, 0, 0 );
		float4 v_t;
		float4 v_n;

		if( x.y <= -halfSpace.y + margin )
		{
			x = make_float4( x.x, -halfSpace.y + margin, x.z, 0 );

			float vDotn = dot( v, n );
			if( vDotn <= 0 )
			{
				v_n = abs(vDotn)*n;
				v_t = v + v_n;
	
				v = restitution*v_n + (1.0f - bFriction)*v_t;
			}
		}
	}

	//@ top wall
	{
		float4 n = make_float4( 0, -1, 0, 0 );
		float4 v_t;
		float4 v_n;

		if( x.y >= halfSpace.y - margin )
		{
			x = make_float4( x.x, halfSpace.y - margin, x.z, 0 );

			float vDotn = dot( v, n );
			if( vDotn <= 0 )
			{
				v_n = abs(vDotn)*n;
				v_t = v + v_n;

				v = restitution*v_n + (1.0f - tFriction)*v_t;
			}
		}
	}

	pPos[index] = x;
	pVel[index] = v;
}

PBFSystem2D::PBFSystem2D()
{
	bPause			= false;
	bMouseActive	= false;
	bSystemReady	= false;
	bDiagnosis		= true;
	bVCOn			= false;
	bXSPHOn			= false;
	bPlot			= false;
}

PBFSystem2D::~PBFSystem2D()
{
	cudaFreeHost(h_d);
	cudaFreeHost(h_v);
	cudaFreeHost(h_p);

	cudaFree(d_d);
	cudaFree(d_lmd);
	cudaFree(d_f);
	cudaFree(d_v);
	cudaFree(d_v_sorted);
	cudaFree(d_p);
	cudaFree(d_p_pred);
	cudaFree(d_p_sorted);
	cudaFree(d_p_sorted_old);
	cudaFree(d_dp);
	cudaFree(d_vx);
	cudaFree(d_cellIndex);
	cudaFree(d_particleIndex);
	cudaFree(d_startCellIndex);
	cudaFree(d_numNeighbors);
	cudaFree(d_neighborIndex);
	cudaFree(d_mat);

	cudaDeviceReset();
}

int fCounter = 0;
void PBFSystem2D::initParticleSystem(SimulationParameters_f4 &pf4)
{
	fCounter = 0;

	cudaDeviceReset();	// added by Min Gyu Choi 2014-12-31

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	bPause			= false;
	bMouseActive	= false;
	bSystemReady	= false;
	bDiagnosis		= true;
	bVCOn			= false;
	bXSPHOn			= false;
	bPlot			= false;

	numParticles	= 0;
	numBlocks		= 0;
	numThreads		= 0;
	substep			= 2;
	numIteration	= 2;

	//@ particle parameters
	float n = 100;													//@ number of particles per unit volume (n/1m^3)
	hSimParams_f.d_rest = 998.2071f;								//@ rest density (kg/m^3)
	hSimParams_f.m		= hSimParams_f.d_rest/n;					//@ particle mass = density*volume/n
	hSimParams_f.h		= 0.34f;									//@ support radius	
	hSimParams_f.gas_k	= 140.0f;									//@ gas constant
	hSimParams_f.visc_u	= 0.001f;									//@ viscosity
	hSimParams_f.cfm_e	= 100.0f;									//@ constraint force mixing
	hSimParams_f.ap_dq	= 0.000f*hSimParams_f.h;					//@ terms for artificial pressure
	hSimParams_f.ap_k	= 0.02f;									//@ ''
	hSimParams_f.ap_n	= 2;										//@ ''
	hSimParams_f.xsph_x	= 0.04f;									//@ xsph constant
	hSimParams_f.vc_e	= 300.0f;									//@ vorticity confinement

	//@ precomputed values
	hSimParams_f.poly6_precomputed = 315.0/(64.0*CUDART_PI_F*pow(hSimParams_f.h, 9.0f));
	hSimParams_f.spiky_precomputed = -45.0/(CUDART_PI_F*pow(hSimParams_f.h, 6.0f));

	//@ grid setting
	hSimParams_f4 = pf4;
	//hSimParams_f.cellSize = 2*hSimParams_f.h;
	hSimParams_f.cellSize = hSimParams_f.h;

	//@ add some workspace margin(1 cell to each walls) for stability...
	hSimParams_f4.workspace = hSimParams_f4.workspace + make_float4(4*hSimParams_f.cellSize, 4*hSimParams_f.cellSize, 4*hSimParams_f.cellSize, 0);
	hSimParams_f4.corner = -0.5f*hSimParams_f4.workspace;

	hSimParams_i.nCellX = ceil(hSimParams_f4.workspace.x/hSimParams_f.cellSize);
	hSimParams_i.nCellY = ceil(hSimParams_f4.workspace.y/hSimParams_f.cellSize);
	hSimParams_i.nCellZ = ceil(hSimParams_f4.workspace.z/hSimParams_f.cellSize);
	hSimParams_i.nCells = hSimParams_i.nCellX*hSimParams_i.nCellY*hSimParams_i.nCellZ;
	
	//@ allocate pinned host memory	
	if( cudaSuccess != cudaMallocHost((void**)&h_d, sizeof(float)*NUM_MAX_PARTICLES) )						{	cout << "Error allocating pinned host memory" << endl; exit(1); }
	if( cudaSuccess != cudaMallocHost((void**)&h_v, sizeof(float4)*NUM_MAX_PARTICLES) )						{	cout << "Error allocating pinned host memory" << endl; exit(1); }
	if( cudaSuccess != cudaMallocHost((void**)&h_p, sizeof(float4)*NUM_MAX_PARTICLES) )						{	cout << "Error allocating pinned host memory" << endl; exit(1); }
	if( cudaSuccess != cudaMallocHost((void**)&h_mat, sizeof(float4)*3*NUM_MAX_PARTICLES) )					{	cout << "Error allocating pinned host memory" << endl; exit(1); }
	
	//@ allocate cuda global memory
	if( cudaSuccess != cudaMalloc(&d_d, sizeof(float)*NUM_MAX_PARTICLES) )									{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_lmd, sizeof(float)*NUM_MAX_PARTICLES) )								{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_f, sizeof(float4)*NUM_MAX_PARTICLES) )									{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_v, sizeof(float4)*NUM_MAX_PARTICLES) )									{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_v_sorted, sizeof(float4)*NUM_MAX_PARTICLES) )							{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_p_sorted_old, sizeof(float4)*NUM_MAX_PARTICLES) )						{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_p, sizeof(float4)*NUM_MAX_PARTICLES) )									{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_p_pred, sizeof(float4)*NUM_MAX_PARTICLES) )							{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_p_sorted, sizeof(float4)*NUM_MAX_PARTICLES) )							{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_dp, sizeof(float4)*NUM_MAX_PARTICLES) )								{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_vx, sizeof(float4)*NUM_MAX_PARTICLES) )								{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_cellIndex, sizeof(uint)*NUM_MAX_PARTICLES) )							{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_particleIndex, sizeof(uint)*NUM_MAX_PARTICLES) )						{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_startCellIndex, sizeof(uint)*(hSimParams_i.nCells)) )					{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_numNeighbors, sizeof(uint)*NUM_MAX_PARTICLES) )						{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_neighborIndex, sizeof(uint)*(NUM_MAX_PARTICLES*NUM_MAX_NEIGHBORS)) )	{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_debug, sizeof(uint)*NUM_MAX_PARTICLES) )								{	cout << "Error allocating cuda device memory" << endl; exit(1); }
	if( cudaSuccess != cudaMalloc(&d_mat, sizeof(float4)*3*NUM_MAX_PARTICLES) )								{	cout << "Error allocating cuda device memory" << endl; exit(1); }

	//@ init device memory
	cudaError_t error;
	for( uint i = 0; i < NUM_MAX_PARTICLES; i++ ) 
		h_d[i] = hSimParams_f.d_rest;
	if( cudaSuccess != (error = cudaMemcpy(d_d, h_d, sizeof(float)*NUM_MAX_PARTICLES, cudaMemcpyHostToDevice)))		{cout << cudaGetErrorString(error) << endl; exit( 1 );}

	if( cudaSuccess != (error = cudaMemset(d_lmd, 0, sizeof(float)*NUM_MAX_PARTICLES)))								{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_f, 0, sizeof(float4)*NUM_MAX_PARTICLES)))								{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_v, 0, sizeof(float4)*NUM_MAX_PARTICLES)))								{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_v_sorted, 0, sizeof(float4)*NUM_MAX_PARTICLES)))						{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_p, 0, sizeof(float4)*NUM_MAX_PARTICLES)))								{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_p_pred, 0, sizeof(float4)*NUM_MAX_PARTICLES)))							{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_p_sorted, 0, sizeof(float4)*NUM_MAX_PARTICLES)))						{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_p_sorted_old, 0, sizeof(float4)*NUM_MAX_PARTICLES)))					{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_dp, 0, sizeof(float4)*NUM_MAX_PARTICLES)))								{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_vx, 0, sizeof(float4)*NUM_MAX_PARTICLES)))								{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_cellIndex, 0, sizeof(uint)*NUM_MAX_PARTICLES)))						{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_particleIndex, 0, sizeof(uint)*NUM_MAX_PARTICLES)))					{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_startCellIndex, 0, sizeof(uint)*(hSimParams_i.nCells))))				{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_numNeighbors, 0, sizeof(uint)*NUM_MAX_PARTICLES)))						{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_neighborIndex, 0, sizeof(uint)*NUM_MAX_PARTICLES*NUM_MAX_NEIGHBORS)))	{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemset(d_mat, 0, sizeof(float4)*3*NUM_MAX_PARTICLES)))							{cout << cudaGetErrorString(error) << endl; exit( 1 );}

	//@ debug
	if( cudaSuccess != (error = cudaMemset(d_debug, 0, sizeof(uint)*NUM_MAX_PARTICLES)))							{cout << cudaGetErrorString(error) << endl; exit( 1 );}

	//@ set constant memory
	if( cudaSuccess != (error = cudaMemcpyToSymbol(cSimParams_f4, &hSimParams_f4, sizeof(SimulationParameters_f4))))	{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemcpyToSymbol(cSimParams_f, &hSimParams_f, sizeof(SimulationParameters_f))))		{cout << cudaGetErrorString(error) << endl; exit( 1 );}
	if( cudaSuccess != (error = cudaMemcpyToSymbol(cSimParams_i, &hSimParams_i, sizeof(SimulationParameters_i))))		{cout << cudaGetErrorString(error) << endl; exit( 1 );}

	//@ diag
	if( bDiagnosis )
	{
		cout << "===== System Information =====" << endl;
		cout << "Size of allocated d-mem: " << (double)NUM_MAX_PARTICLES*(sizeof(uint)*NUM_MAX_NEIGHBORS + sizeof(uint)*5 + sizeof(float) + sizeof(float4)*7)/(1024*1024) << "MB" << endl;
		cout << "Number of particles/v  : " << n << endl; 
		cout << "Rest density           : " << hSimParams_f.d_rest << endl;
		cout << "Particle mass          : " << hSimParams_f.m << endl;
		cout << "Support radius         : " << hSimParams_f.h << endl;
		cout << "Workspace              : " << hSimParams_f4.workspace.x << "m x " << hSimParams_f4.workspace.y << "m x " << hSimParams_f4.workspace.z << "m" << endl; 
		cout << "Spatial grid size      : " << hSimParams_i.nCellX << " x " << hSimParams_i.nCellY << " x " << hSimParams_i.nCellZ << " = " << hSimParams_i.nCells << endl;
		cout << "==============================" << endl;
	}

	bSystemReady = true;
}

void PBFSystem2D::createParticle(float4 &pos, float4 &vel)
{
	if( numParticles == NUM_MAX_PARTICLES ) return;

	if( pos.x < hSimParams_f4.corner.x || pos.x > hSimParams_f4.corner.x + hSimParams_f4.workspace.x || 
		pos.y < hSimParams_f4.corner.y || pos.y > hSimParams_f4.corner.y + hSimParams_f4.workspace.y )
		return;

	numParticles++;
	g_createParticle<<<1,1>>>(numParticles - 1, vel, pos, d_v, d_p);
}

void PBFSystem2D::createParticleNbyN(uint n, float spacing, float4 pos, float4 vel)
{
	if( numParticles >= NUM_MAX_PARTICLES ) return;

	uint numParticleToEmit = n*n;
	if( numParticleToEmit + numParticles > NUM_MAX_PARTICLES )
		numParticleToEmit = NUM_MAX_PARTICLES - numParticles;

	uint k = sqrt(numParticleToEmit);
	numParticleToEmit = k*k;

	g_createParticleNbyN<<<1, 1>>>(numParticles, k, k, spacing, vel, pos, d_v, d_p);
	cudaDeviceSynchronize();

	numParticles += numParticleToEmit;
}

int counter = 0;
clock_t t_totalSim = 0;
clock_t t_pred = 0;
clock_t t_neighbor = 0;
clock_t t_solve = 0;
clock_t t_update = 0;
clock_t t_xsph = 0;
clock_t t_vc = 0;
void PBFSystem2D::simulate(float dt)
{
	fCounter++;

	float dt_step	= dt/substep;

	clock_t start; 
	clock_t end;

	clock_t step_start;
	clock_t step_end;

	start = clock();
	for(uint step = 0; step < substep; step++)
	{
		step_start = clock();
		prediction(dt_step);
		step_end = clock();
		t_pred += step_end - step_start;

		step_start = clock();
		neighborSearch();
		step_end = clock();
		t_neighbor += step_end - step_start;
		
		step_start = clock();
		for(uint iter = 0; iter < numIteration; iter++)
			solve(dt_step);
		step_end = clock();
		t_solve += step_end - step_start;

		step_start = clock();
		update(dt_step);
		step_end = clock();
		t_update += step_end - step_start;

		step_start = clock();
		xsph_vorticity();
		step_end = clock();
		t_xsph += step_end - step_start;

		step_start = clock();
		vorticityConfinement();
		step_end = clock();
		t_vc += step_end - step_start;

		collisionResponse();
	}
	end = clock();
	t_totalSim += end - start;
	counter++;

	//cout << counter << ": " << (float)t_totalSim/counter << ", " << (float)t_pred/counter << ", " << (float)t_neighbor/counter << ", " 
	//	<< (float)t_solve/counter << ", " << (float)t_update/counter << ", " << (float)t_xsph/counter << ", " << (float)t_vc/counter << endl;
	//cout << counter << ": " << (float)t_totalSim << ", " << t_pred << ", " << t_neighbor << ", " << t_solve << ", " << t_update << ", " << t_xsph << ", " << t_vc << endl;

	t_totalSim = 0;
	t_pred = 0;
	t_neighbor = 0;
	t_solve = 0;
	t_update = 0;
	t_xsph = 0;
	t_vc = 0;

	cudaMemcpy(h_p, d_p, numParticles*sizeof(float4), cudaMemcpyDeviceToHost); 
	//cudaMemcpy(h_mat, d_mat, numParticles*3*sizeof(float4), cudaMemcpyDeviceToHost); 
}

void PBFSystem2D::prediction(float dt)
{
	cudaBindTexture(0, tex_pos, d_p, sizeof(float4)*numParticles);
	cudaBindTexture(0, tex_vel, d_v, sizeof(float4)*numParticles);
	cudaBindTexture(0, tex_density, d_d, sizeof(float)*numParticles);

	numThreads	= 256;
	numBlocks	= numParticles/numThreads + 1;
	setDeviceUintArray<<<numBlocks, numThreads>>>(numParticles, true, 0, d_particleIndex);
	cudaMemset(d_startCellIndex, 0, sizeof(uint)*(hSimParams_i.nCells));
	cudaMemset(d_numNeighbors, 0, sizeof(uint)*numParticles);
	cudaMemset(d_cellIndex, 0, sizeof(uint)*numParticles);

	numThreads	= 64;
	numBlocks	= numParticles/numThreads + 1;
	g_prediction<<<numBlocks, numThreads>>>(numParticles, d_p, d_d, dt, d_f, d_v, d_p_pred, d_cellIndex);
	cudaDeviceSynchronize();

	cudaUnbindTexture(tex_pos);
	cudaUnbindTexture(tex_vel);
	cudaUnbindTexture(tex_density);
}

void PBFSystem2D::modifiedPrediction(float dt)
{
}
 
int maxNeighbor = 0;
int maxnPerCell = 0;
void PBFSystem2D::neighborSearch()
{
	cudaBindTexture(0, tex_sortedPos, d_p_sorted, sizeof(float4)*numParticles);
	cudaBindTexture(0, tex_cellIndex, d_cellIndex, sizeof(uint)*numParticles);

	thrust::sort_by_key(thrust::device_ptr<uint>(d_cellIndex), thrust::device_ptr<uint>(d_cellIndex + numParticles), thrust::device_ptr<uint>(d_particleIndex));
	
	numThreads	= 256;
	numBlocks	= numParticles/numThreads + 1;
	g_findStartCellIndex<<<numBlocks, numThreads>>>(numParticles, d_cellIndex, d_particleIndex, d_p_pred, d_startCellIndex, d_p_sorted, d_p_sorted_old);
	cudaDeviceSynchronize();

	uint particlesPerThread = 1;
	numThreads	= 32*16;
	numBlocks	= numParticles/(numThreads*particlesPerThread) + 1;
	g_setNeighborList<<<numBlocks, numThreads>>>(numParticles, particlesPerThread, d_p_sorted, d_cellIndex, d_startCellIndex, d_particleIndex, d_numNeighbors, d_neighborIndex);
	cudaDeviceSynchronize();

	cudaUnbindTexture(tex_sortedPos);
	cudaUnbindTexture(tex_cellIndex);
}	

void PBFSystem2D::solve(float dt)
{
	cudaBindTexture(0, tex_sortedPos, d_p_sorted, sizeof(float4)*numParticles);
	cudaBindTexture(0, tex_neighborIndex, d_neighborIndex, sizeof(uint)*numParticles*NUM_MAX_NEIGHBORS);

	numThreads	= 64;
	numBlocks	= numParticles/numThreads + 1;
	g_computeLambda<<<numBlocks, numThreads>>>(numParticles, d_p_sorted, d_numNeighbors, d_neighborIndex, d_d, d_lmd);
	g_computePositionDelta<<<numBlocks, numThreads>>>(numParticles, d_p_sorted, d_lmd, d_numNeighbors, d_neighborIndex, d_dp);

	numThreads	= 256;
	numBlocks	= numParticles/numThreads + 1;
	g_correction<<<numBlocks, numThreads>>>(numParticles, d_particleIndex, d_p_pred, d_p_sorted, d_dp);
	cudaDeviceSynchronize();

	cudaUnbindTexture(tex_sortedPos);
	cudaUnbindTexture(tex_neighborIndex);
}

void PBFSystem2D::update(float dt)
{
	cudaBindTexture(0, tex_sortedPos, d_p_sorted, sizeof(float4)*numParticles);
	cudaBindTexture(0, tex_neighborIndex, d_neighborIndex, sizeof(uint)*numParticles*NUM_MAX_NEIGHBORS);

	numThreads	= 64;
	numBlocks	= numParticles/numThreads + 1;
	g_update<<<numBlocks, numThreads>>>(numParticles, d_p_pred, d_p_sorted_old, d_particleIndex, dt, d_v, d_v_sorted, d_p);
	cudaDeviceSynchronize();

	//g_anisotrophy<<<numBlocks, numThreads>>>(numParticles, d_numNeighbors, d_particleIndex, d_mat); 
	//cudaDeviceSynchronize();

	cudaUnbindTexture(tex_sortedPos);
	cudaUnbindTexture(tex_neighborIndex);
}

void PBFSystem2D::xsph_vorticity()
{
	cudaBindTexture(0, tex_sortedPos, d_p_sorted, sizeof(float4)*numParticles);
	cudaBindTexture(0, tex_sortedVel, d_v_sorted, sizeof(float4)*numParticles);
	cudaBindTexture(0, tex_neighborIndex, d_neighborIndex, sizeof(uint)*numParticles*NUM_MAX_NEIGHBORS);

	numThreads	= 64;
	numBlocks	= numParticles/numThreads + 1;
	g_xsph_vorticity<<<numBlocks, numThreads>>>(numParticles, d_p_sorted, d_d, d_numNeighbors, d_particleIndex, d_v, d_vx);
	cudaDeviceSynchronize();

	cudaUnbindTexture(tex_sortedPos);
	cudaUnbindTexture(tex_sortedVel);
	cudaUnbindTexture(tex_neighborIndex);
}

void PBFSystem2D::vorticityConfinement()
{
	cudaBindTexture(0, tex_sortedPos, d_p_sorted, sizeof(float4)*numParticles);
	cudaBindTexture(0, tex_neighborIndex, d_neighborIndex, sizeof(uint)*numParticles*NUM_MAX_NEIGHBORS);

	numThreads	= 64;
	numBlocks	= numParticles/numThreads + 1;
	g_vorticityConfinement<<<numBlocks, numThreads>>>(numParticles, d_p_sorted, d_d, d_vx, d_numNeighbors, d_particleIndex, d_f);
	cudaDeviceSynchronize();

	cudaUnbindTexture(tex_sortedPos);
	cudaUnbindTexture(tex_neighborIndex);
}

void PBFSystem2D::collisionResponse()
{
	cudaBindTexture(0, tex_pos, d_p, sizeof(float4)*numParticles);
	cudaBindTexture(0, tex_vel, d_v, sizeof(float4)*numParticles);

	numThreads	= 256;
	numBlocks	= numParticles/numThreads + 1;
	g_wallCollision<<<numBlocks, numThreads>>>(numParticles, d_p, d_v);
	cudaDeviceSynchronize();

	cudaUnbindTexture(tex_pos);
	cudaUnbindTexture(tex_vel);
}

