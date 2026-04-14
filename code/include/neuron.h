#ifndef NEURON
#define NEURON

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define IR (0.01)
#define LR (0.1)

typedef struct neuron {
    float a;
    float xh; //x hat, prediction of x
    float x; //latent value or input
    float err; //error
} Neuron;

typedef struct layer {
    int numUpper;
    int numLower;
    Neuron* lower;
    float* weights; //nullptr if top layer
} Layer;

typedef struct network {
    int numLayers;
    Layer* layers;
} Network;

void initNeuron(Neuron* n);
void initLayer(int numUpper, int numLower, Layer* layer);
void initNetwork(int* layerSizes, int numLayers, Network* network);

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

void evaluateNetwork(Network* network, float* inputs, float* output, int numIters);

#endif