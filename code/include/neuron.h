#ifndef NEURON
#define NEURON

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define IR (0.01f)
#define LR (0.1f)

#define RMIN (0.4f)
#define RRANGE (0.2f)

typedef struct {
    float a;
    float xh; //x hat, prediction of x
    float x; //latent value or input
    float err; //error
} Neuron;

typedef struct {
    int numUpper;
    int numLower;
    Neuron* lower;
    float* weights; //nullptr if top layer
} Layer;

typedef struct {
    int numLayers;
    Layer* layers;
} Network;

typedef struct {
    int numLayers;
    int numNeurons;
    int numWeights;
    Neuron* neurons;
    int* layerSizes;
    int* layerOffsets;
    int* weightOffsets;
    float* weights;

    void* randStates;
} NetworkGPU;

void initNeuron(Neuron* n);
void initLayer(int numUpper, int numLower, Layer* layer);
void initNetwork(int* layerSizes, int numLayers, Network* network);

void freeLayer(Layer* layer);
void freeNetwork(Network* network);

void updateLayer(Layer* layer, Layer* nextLayer);
void updateLayerInference(Layer* layer, Layer* prevLayer);
void updateInputLayerInference(Layer* layer);
void updateLayerWeights(Layer* layer, Layer* nextLayer);

void randomizeLayerLatents(Layer* layer);

void updateNetwork(Network* network);
void updateNetworkInference(Network* network, char updateInput);
void updateNetworkWeights(Network* network);

void setNetworkInputs(Network* network, float* inputs);
void setNetworkOutputs(Network* network, float* outputs);
void randomizeNetworkLatents(Network* network);

float getLoss(Network* network);

void printLayer(Layer* layer);
void printNetwork(Network* network);

void trainNetwork(Network* network, float** inputs, int numSamples, int numLearnIters, int numInferIters);
void evaluateNetwork(Network* network, float* inputs, float* output, int numIters);
void generateOutput(Network* network, int numIters);

//GPU stuff

void freeNetworkGPU(NetworkGPU network);
void copyNetworkToGPU(Network* from, NetworkGPU* to);
void copyNetworkFromGPU(NetworkGPU from, Network* to);

void setNetworkInputsGPU(NetworkGPU network, float* inputs);
void setNetworkOutputsGPU(NetworkGPU network, float* outputs);
void randomizeNetworkLatentsGPU(NetworkGPU network);

void trainNetworkGPU(NetworkGPU network, float** inputs, int numSamples, int numLearnIters, int numInferIters);
void evaluateNetworkGPU(NetworkGPU network, float* inputs, float* output, int numIters);
void generateOutputGPU(NetworkGPU network, int numIters);

#endif
