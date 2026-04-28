extern "C" {
#include "neuron.h"
}
#include "curand_kernel.h"
#include <cstdio>
#include <cstdlib>

__device__ float f_d(float x) {
	return 1.0f / (1.0f + exp(-x));
}

__device__ float dfdx_d(float x) {
	float enx = exp(-x);
	return enx / ((1 + enx) * (1 + enx));
}

__global__ void updateNetworkGPU(NetworkGPU* network) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= network->numNeurons) {
		return;
	}

	int layerIndex = 0;
	for(int j = 0; j < network->numLayers; j++) {
		if(i >= network->layerOffsets[j]) {
			layerIndex = j;
			break;
		}
	}
	int inLayerIndex = i - network->layerOffsets[layerIndex];
	
	network->neurons[i].a = 0;
	if(layerIndex < network->numLayers - 1) {
		for(int j = 0; j < network->layerSizes[layerIndex + 1]; j++) {
			network->neurons[i].a += network->neurons[network->layerOffsets[layerIndex + 1] + j].x * 
					network->weights[network->weightOffsets[layerIndex] + inLayerIndex * network->layerSizes[layerIndex + 1] + j];
		}
	}

	network->neurons[i].xh = f_d(network->neurons[i].a);
	network->neurons[i].err = network->neurons[i].x - network->neurons[i].xh;
}

__global__ void updateNetworkInferenceGPU(NetworkGPU* network, char updateInputs) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= network->numNeurons) {
		return;
	}

	int layerIndex = 0;
	for(int j = 0; j < network->numLayers; j++) {
		if(i >= network->layerOffsets[j]) {
			layerIndex = j;
			break;
		}
	}
	int inLayerIndex = i - network->layerOffsets[layerIndex];

	if(layerIndex > 0) {
		float n = 0;
		for(int j = 0; j < network->layerSizes[layerIndex - 1]; j++) {
			float h = dfdx_d(network->neurons[network->layerOffsets[layerIndex - 1] + j].a) * network->neurons[network->layerOffsets[layerIndex - 1] + j].err;
			n += network->weights[network->weightOffsets[layerIndex - 1] + j * network->layerSizes[layerIndex] + inLayerIndex] * h;
		}
		network->neurons[i].x -= IR * (network->neurons[i].err - n);
	} else if(updateInputs) {
		network->neurons[i].x -= IR * (network->neurons[i].err);
	}
}

__global__ void updateNetworkWeightsGPU(NetworkGPU* network) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= network->numNeurons) {
		return;
	}

	int layerIndex = 0;
	for(int j = 0; j < network->numLayers; j++) {
		if(i >= network->layerOffsets[j]) {
			layerIndex = j;
			break;
		}
	}
	int inLayerIndex = i - network->layerOffsets[layerIndex];

	if(layerIndex < network->numLayers - 1) {
		for(int j = 0; j < network->layerSizes[layerIndex + 1]; j++) {
			float h = dfdx_d(network->neurons[network->layerOffsets[layerIndex] + j].a) * network->neurons[network->layerOffsets[layerIndex] + j].err;
			network->weights[network->weightOffsets[layerIndex] + inLayerIndex * network->layerSizes[layerIndex + 1] + j] +=
					LR * h * network->neurons[network->layerOffsets[layerIndex + 1] + j].x;
		}
	}
}


__global__ void initCurand(curandState* states, int numStates) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i >= numStates) {
		return;
	}
	
	curand_init(1234, i, 0, &states[i]);
}

__global__ void __randomizeNetworkLatentsGPU(NetworkGPU *network) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i >= network->numNeurons) {
		return;
	}

	network->neurons[i].x = curand_uniform(&((curandState*) network->randStates)[i]) * RRANGE + RMIN;
}

__global__ void inferenceIters(NetworkGPU* network, int numIters, char updateInputs) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= network->numNeurons) {
		return;
	}

	int layerIndex = 0;
	for(int j = 0; j < network->numLayers; j++) {
		if(i >= network->layerOffsets[j]) {
			layerIndex = j;
			break;
		}
	}
	int inLayerIndex = i - network->layerOffsets[layerIndex];

	__syncthreads();
	for(int k = 0; k < numIters; k++) {
		//update network
		network->neurons[i].a = 0;
		if(layerIndex < network->numLayers - 1) {
			for(int j = 0; j < network->layerSizes[layerIndex + 1]; j++) {
				network->neurons[i].a += network->neurons[network->layerOffsets[layerIndex + 1] + j].x * 
						network->weights[network->weightOffsets[layerIndex] + inLayerIndex * network->layerSizes[layerIndex + 1] + j];
			}
		}

		network->neurons[i].xh = f_d(network->neurons[i].a);
		network->neurons[i].err = network->neurons[i].x - network->neurons[i].xh;

		__syncthreads();

		//update inference
		if(layerIndex > 0) {
			float n = 0;
			for(int j = 0; j < network->layerSizes[layerIndex - 1]; j++) {
				float h = dfdx_d(network->neurons[network->layerOffsets[layerIndex - 1] + j].a) * network->neurons[network->layerOffsets[layerIndex - 1] + j].err;
				n += network->weights[network->weightOffsets[layerIndex - 1] + j * network->layerSizes[layerIndex] + inLayerIndex] * h;
			}
			network->neurons[i].x -= IR * (network->neurons[i].err - n);
		} else if(updateInputs) {
			network->neurons[i].x -= IR * (network->neurons[i].err);
		}

		__syncthreads();
	}
}

extern "C"
NetworkGPU* createNetworkGPU() {
	NetworkGPU* ret;
	cudaMalloc(&ret, sizeof(NetworkGPU));
	return ret;
}

extern "C"
void freeNetworkGPU(NetworkGPU* network) {
	cudaFree(network->neurons);
	cudaFree(network->weights);
	cudaFree(network->weightOffsets);
	cudaFree(network->layerSizes);
	cudaFree(network->layerOffsets);
	cudaFree(network->randStates);
	cudaFree(network);
}

extern "C"
void copyNetworkToGPU(Network* from, NetworkGPU* to) {
	int totalNeurons = 0;
	int totalWeights = 0;
	NetworkGPU to_h;
	to_h.layerSizes = (int*) malloc(sizeof(int) * from->numLayers);
	to_h.layerOffsets = (int*) malloc(sizeof(int) * from->numLayers);
	to_h.weightOffsets = (int*) malloc(sizeof(int) * from->numLayers);
	to_h.layerOffsets[0] = 0;
	to_h.weightOffsets[0] = 0;
	for(int i = 0; i < from->numLayers - 1; i++) {
		Layer l = from->layers[i];
		to_h.layerSizes[i] = l.numLower;
		to_h.layerOffsets[i + 1] = to_h.layerOffsets[i] + l.numLower;
		to_h.weightOffsets[i + 1] = to_h.weightOffsets[i] + l.numLower * l.numUpper;
		totalNeurons += l.numLower;
		totalWeights += l.numLower * l.numUpper;
	}
	to_h.layerSizes[from->numLayers - 1] = from->layers[from->numLayers - 1].numLower;
	totalNeurons += from->layers[from->numLayers - 1].numLower;
	//top layer has zero weights
	cudaMalloc(&to->neurons, sizeof(Neuron) * totalNeurons);
	cudaMalloc(&to->weights, sizeof(float) * totalWeights);
	for(int i = 0; i < from->numLayers; i++) {
		Layer l = from->layers[i];
		for(int j = 0; j < l.numLower; i++) {
			to_h.neurons[to_h.layerOffsets[i] + j] = l.lower[i];
			for(int k = 0; k < l.numUpper; k++) {
				to_h.weights[to_h.weightOffsets[i] + i * l.numUpper + j] = l.weights[i * l.numUpper + j];
			}
		}
		cudaMemcpy(to->neurons, &l.lower[to_h.layerOffsets[i]], sizeof(Neuron) * to_h.layerSizes[i], cudaMemcpyHostToDevice);
		if(i < from->numLayers - 1) {
			cudaMemcpy(to->weights, &l.weights[to_h.weightOffsets[i]], sizeof(float) * to_h.layerSizes[i] * to_h.layerSizes[i + 1], cudaMemcpyHostToDevice);
		}
	}
	cudaMalloc(&to->layerSizes, sizeof(int) * from->numLayers);
	cudaMalloc(&to->layerOffsets, sizeof(int) * from->numLayers);
	cudaMalloc(&to->weightOffsets, sizeof(int) * from->numLayers);

	cudaMemcpy(to->layerSizes, to_h.layerSizes, sizeof(int) * from->numLayers, cudaMemcpyHostToDevice);
	cudaMemcpy(to->layerOffsets, to_h.layerOffsets, sizeof(int) * from->numLayers, cudaMemcpyHostToDevice);
	cudaMemcpy(to->weightOffsets, to_h.weightOffsets, sizeof(int) * from->numLayers, cudaMemcpyHostToDevice);
	cudaMemcpy(to->neurons, to_h.neurons, sizeof(float) * totalNeurons, cudaMemcpyHostToDevice);
	cudaMemcpy(to->weights, to_h.weights, sizeof(float) * totalWeights, cudaMemcpyHostToDevice);

	cudaMemcpy(&to->numLayers, &from->numLayers, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&to->numNeurons, &totalNeurons, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&to->numWeights, &totalWeights, sizeof(int), cudaMemcpyHostToDevice);

	//maybe this shouldnt be in here idk
	cudaMalloc(&to->randStates, sizeof(curandState) * to->numNeurons);
	dim3 grid = dim3((to->numNeurons + 511) / 512);
	dim3 block = dim3(512);
	initCurand<<<grid, block>>>((curandState*) to->randStates, to->numNeurons);

	free(to_h.layerOffsets);
	free(to_h.layerSizes);
	free(to_h.weightOffsets);
}

//assumes receiving network is already allocated and has correct shape
extern "C"
void copyNetworkFromGPU(NetworkGPU* from, Network* to) {
	for(int i = 0; i < to->numLayers; i++) {
		cudaMemcpy(to->layers[i].lower, &from->neurons[from->layerOffsets[i]], 
			sizeof(Neuron) * to->layers[i].numLower, cudaMemcpyDeviceToHost);
		cudaMemcpy(to->layers[i].weights, &from->weights[from->weightOffsets[i]], 
			sizeof(float) * to->layers[i].numLower * to->layers[i].numUpper, cudaMemcpyDeviceToHost);
	}
}

extern "C"
void trainNetworkGPU(NetworkGPU* network, float** inputs, int numSamples, int numLearnIters, int numInferIters) {

}

extern "C"
void evaluateNetworkGPU(NetworkGPU* network, float* inputs, float* outputs, int numIters) {

}

extern "C"
void generateOutputGPU(NetworkGPU* network, int numIters) {
	dim3 grid = dim3((network->numNeurons + 511) / 512);
	dim3 block = dim3(512);
	__randomizeNetworkLatentsGPU<<<grid, block>>>(network);
	for(int i = 0; i < 10; i++) {
		inferenceIters<<<grid, block>>>(network, numIters / 10, 1);
		printf("\rGenerating: %d%%", i * 10);
	}
	printf("Done Generating!\n");
}