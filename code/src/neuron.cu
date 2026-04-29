extern "C" {
#include "neuron.h"
}
#include "curand_kernel.h"
#include <cstdlib>

__device__ float f_d(float x) {
	return 1.0f / (1.0f + exp(-x));
	// return x;
}

__device__ float dfdx_d(float x) {
	float enx = expf(-x);
	return enx / ((1 + enx) * (1 + enx));
}

__global__ void updateNetworkGPU(NetworkGPU network) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= network.numNeurons) {
		return;
	}

	int layerIndex = 0;
	for(int j = network.numLayers - 1; j >= 0; j--) {
		if(i >= network.layerOffsets[j]) {
			layerIndex = j;
			break;
		}
	}
	int inLayerIndex = i - network.layerOffsets[layerIndex];

	network.neurons[i].a = 0;
	if(layerIndex < network.numLayers - 1) {
		for(int j = 0; j < network.layerSizes[layerIndex + 1]; j++) {
			int nIndex = network.layerOffsets[layerIndex + 1] + j;
			int wIndex = network.weightOffsets[layerIndex] + inLayerIndex * network.layerSizes[layerIndex + 1] + j;
			float cont = network.neurons[nIndex].x * network.weights[wIndex];
			network.neurons[i].a += cont;
		}
	}

	network.neurons[i].xh = f_d(network.neurons[i].a);
	network.neurons[i].err = network.neurons[i].x - network.neurons[i].xh;
}

__global__ void updateNetworkInferenceGPU(NetworkGPU network, char updateInputs) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= network.numNeurons) {
		return;
	}

	int layerIndex = 0;
	for(int j = network.numLayers - 1; j >= 0; j--) {
		if(i >= network.layerOffsets[j]) {
			layerIndex = j;
			break;
		}
	}
	int inLayerIndex = i - network.layerOffsets[layerIndex];

	if(layerIndex > 0) {
		float n = 0;
		for(int j = 0; j < network.layerSizes[layerIndex - 1]; j++) {
			float h = dfdx_d(network.neurons[network.layerOffsets[layerIndex - 1] + j].a) * network.neurons[network.layerOffsets[layerIndex - 1] + j].err;
			n += network.weights[network.weightOffsets[layerIndex - 1] + j * network.layerSizes[layerIndex] + inLayerIndex] * h;
		}
		network.neurons[i].x -= IR * (network.neurons[i].err - n);
	} else if(updateInputs) {
		network.neurons[i].x -= IR * (network.neurons[i].err);
	}
}

__global__ void updateNetworkWeightsGPU(NetworkGPU network) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= network.numNeurons) {
		return;
	}

	int layerIndex = 0;
	for(int j = network.numLayers - 1; j >= 0; j--) {
		if(i >= network.layerOffsets[j]) {
			layerIndex = j;
			break;
		}
	}
	int inLayerIndex = i - network.layerOffsets[layerIndex];

	if(layerIndex < network.numLayers - 1) {
		for(int j = 0; j < network.layerSizes[layerIndex + 1]; j++) {
			float h = dfdx_d(network.neurons[network.layerOffsets[layerIndex] + j].a) * network.neurons[network.layerOffsets[layerIndex] + j].err;
			network.weights[network.weightOffsets[layerIndex] + inLayerIndex * network.layerSizes[layerIndex + 1] + j] +=
					LR * h * network.neurons[network.layerOffsets[layerIndex + 1] + j].x;
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

__global__ void __randomizeNetworkLatentsGPU(NetworkGPU network) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i >= network.numNeurons) {
		return;
	}

	network.neurons[i].x = curand_uniform(&((curandState*) network.randStates)[i]) * RRANGE + RMIN;
}

__global__ void __setNetworkInputs(NetworkGPU network, float* input) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i >= network.layerSizes[0]) {
		return;
	}

	network.neurons[i].x = input[i];
}

extern "C"
void freeNetworkGPU(NetworkGPU network) {
	cudaFree(network.neurons);
	cudaFree(network.weights);
	cudaFree(network.weightOffsets);
	cudaFree(network.layerSizes);
	cudaFree(network.layerOffsets);
	cudaFree(network.randStates);
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
	cudaError_t e = cudaMalloc(&to->neurons, sizeof(Neuron) * totalNeurons);
	e = cudaMalloc(&to->weights, sizeof(float) * totalWeights);
	for(int i = 0; i < from->numLayers; i++) {
		Layer l = from->layers[i];
		e = cudaMemcpy(to->neurons + to_h.layerOffsets[i], l.lower, sizeof(Neuron) * to_h.layerSizes[i], cudaMemcpyHostToDevice);
		if(i < from->numLayers - 1) {
			//WHY IS IT SEGFAULTING HERE IM GOING INSANE
			e = cudaMemcpy(to->weights + to_h.weightOffsets[i], l.weights, sizeof(float) * to_h.layerSizes[i] * to_h.layerSizes[i + 1], cudaMemcpyHostToDevice);
		}
	}

	cudaMalloc(&to->layerSizes, sizeof(int) * from->numLayers);
	cudaMalloc(&to->layerOffsets, sizeof(int) * from->numLayers);
	cudaMalloc(&to->weightOffsets, sizeof(int) * from->numLayers);

	cudaMemcpy(to->layerSizes, to_h.layerSizes, sizeof(int) * from->numLayers, cudaMemcpyHostToDevice);
	cudaMemcpy(to->layerOffsets, to_h.layerOffsets, sizeof(int) * from->numLayers, cudaMemcpyHostToDevice);
	cudaMemcpy(to->weightOffsets, to_h.weightOffsets, sizeof(int) * from->numLayers, cudaMemcpyHostToDevice);

	to->numLayers = from->numLayers;
	to->numNeurons = totalNeurons;
	to->numWeights = totalWeights;

	//maybe this shouldnt be in here idk
	cudaMalloc(&to->randStates, sizeof(curandState) * to->numNeurons);
	dim3 grid = dim3((to->numNeurons + 511) / 512);
	dim3 block = dim3(512);
	initCurand<<<grid, block>>>((curandState*) to->randStates, to->numNeurons);
	cudaDeviceSynchronize();

	free(to_h.layerOffsets);
	free(to_h.layerSizes);
	free(to_h.weightOffsets);
}

//assumes receiving network is already allocated and has correct shape
extern "C"
void copyNetworkFromGPU(NetworkGPU from, Network* to) {
	unsigned int neuronIndex = 0;
	unsigned int weightIndex = 0;

	for(int i = 0; i < to->numLayers; i++) {
		cudaMemcpy(to->layers[i].lower, from.neurons + neuronIndex, 
			sizeof(Neuron) * to->layers[i].numLower, cudaMemcpyDeviceToHost);
		cudaMemcpy(to->layers[i].weights, from.weights + weightIndex, 
			sizeof(float) * to->layers[i].numLower * to->layers[i].numUpper, cudaMemcpyDeviceToHost);
		// memset(to->layers[i].lower, 0, sizeof(Neuron) * to->layers[i].numLower);
		neuronIndex += to->layers[i].numLower;
		weightIndex += to->layers[i].numLower * to->layers[i].numUpper;
	}
}

// void setNetworkInputsGPU(NetworkGPU network, float* inputs) {

// }

// void setNetworkOutputsGPU(NetworkGPU network, float* outputs) {

// }

extern "C"
void trainNetworkGPU(NetworkGPU network, float** inputs, int numSamples, int numLearnIters, int numInferIters) {
	dim3 grid = dim3((network.numNeurons + 511) / 512);
	dim3 block = dim3(512);
	for(int i = 0; i < numLearnIters; i++) {
		int c = rand() % numSamples;
		__randomizeNetworkLatentsGPU<<<grid, block>>>(network);
		cudaDeviceSynchronize();
		for(int j = 0; j < numInferIters; j++) {
			__setNetworkInputs<<<grid, block>>>(network, inputs[c]);
			cudaDeviceSynchronize();
			updateNetworkGPU<<<grid, block>>>(network);
			cudaDeviceSynchronize();
			updateNetworkInferenceGPU<<<grid, block>>>(network, 0);
			cudaDeviceSynchronize();
		}
		//test if this is needed
		__setNetworkInputs<<<grid, block>>>(network, inputs[c]);
		cudaDeviceSynchronize();
		updateNetworkGPU<<<grid, block>>>(network);
		cudaDeviceSynchronize();
		updateNetworkWeightsGPU<<<grid, block>>>(network);
		cudaDeviceSynchronize();
	}
}

extern "C"
void evaluateNetworkGPU(NetworkGPU network, float* inputs, float* outputs, int numIters) {

}

extern "C"
void generateOutputGPU(NetworkGPU network, int numIters) {
	dim3 grid = dim3((network.numNeurons + 511) / 512);
	dim3 block = dim3(512);
	__randomizeNetworkLatentsGPU<<<grid, block>>>(network);
	cudaDeviceSynchronize();
	for(int i = 0; i < numIters; i++) {
		// printf("\rGenerating: %5d/%d", i + 1, numIters);
		updateNetworkGPU<<<grid, block>>>(network);
		cudaDeviceSynchronize();

		updateNetworkInferenceGPU<<<grid, block>>>(network, 1);
		cudaDeviceSynchronize();
	}
}
