#version 330


uniform sampler2D tex;
uniform vec2 viewport;
uniform float scale;
uniform float tonalScale;

in vec2 texCoord;

layout (location = 0) out vec4 fragColor;
layout (location = 1) out float depth;

float PI = 3.141592653f;
float gaussianFunction(float t, float sigma)
{
	return (1.0f/(sigma*sqrt(2.0f*PI)))*exp((-t*t)/(2.0f*sigma*sigma));
}

// returns filtered depth [0-1] 
float bilateralFilter(vec2 coord, vec2 texelSize)
{
	float depth = texture(tex, coord).x;
	
	float weightedSampleSum = 0; 
	float weightSum = 0;

	int numXSamples = int(scale*100*pow(1.0f - depth,2.0f) + 14);
	//int numXSamples = int(scale*100*pow(1.0f - depth,2.0f) + 24);

	//float spatialScale = numXSamples*texelSize.x/3.0f;
	float spatialScale = 0.008;
	//float tonalScale = 0.025f/3.0f;

	for( int i = -numXSamples; i <= numXSamples; i++ )
	{
		vec2 sampleCoord = texCoord + vec2(i*texelSize.x, 0);
		if( sampleCoord.x < 0 || sampleCoord.y < 0 || sampleCoord.x > 1.0 || sampleCoord.y > 1.0 )
			continue;

		float sample = texture(tex, sampleCoord).x;

		if( sample > 0.99 ) continue;

		float spatialWeight = gaussianFunction(length(sampleCoord - texCoord), spatialScale);
		float tonalWeight	= gaussianFunction(sample - depth, tonalScale);
		float bilateralWeight = spatialWeight*tonalWeight;

		weightedSampleSum += bilateralWeight*sample;
		weightSum += bilateralWeight;
	}

	if(weightSum > 0.0)
		weightedSampleSum = weightedSampleSum/weightSum;

	return weightedSampleSum;
}

void main()
{
	vec2 texelSize = vec2(1.0/viewport.x, 1.0/viewport.y);
	float texDepth = texture(tex, texCoord).x;
	float fDepth = bilateralFilter(texCoord, texelSize);
	if( texDepth > 0.99 || fDepth > 0.99 )
	{ 
		//depth = 1.0;
		//gl_FragDepth = 1.0;
		discard;
	}
	
	else
	{
		fragColor = vec4(0.0, 0.0, 0.0, 1.0);
		depth = fDepth;
		gl_FragDepth = fDepth;
	}
}
