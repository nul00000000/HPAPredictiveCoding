#ifndef NEURON
#define NEURON

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define IR (0.01)
#define LR (0.01)

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
void updateLayerWeights(Layer* layer, Layer* nextLayer);

void updateNetwork(Network* network);
void updateNetworkInference(Network* network);
void updateNetworkWeights(Network* network);

void setNetworkInputs(Network* network, float* inputs);
void setNetworkOutputs(Network* network, float* outputs);

float getLoss(Network* network);

void printLayer(Layer* layer);
void printNetwork(Network* network);

void evaluateNetwork(Network* network, float* inputs, float* output, int numIters);

#endif