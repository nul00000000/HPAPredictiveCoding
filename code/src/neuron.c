#include "neuron.h"
#include "stdlib.h"

void initLayer(int numUpper, int numLower, Layer* layer) {
    layer->numUpper = numUpper;
    layer->numLower = numLower;
    layer->upper = (Neuron*) malloc(numUpper * sizeof(Neuron));
    if(numUpper * numLower > 0) {
        layer->weights = (float*) malloc(numUpper * numLower * sizeof(float));
        //maybe randomize in another function?
        for(int i = 0; i < numUpper * numLower; i++) {
            layer->weights[i] = (float) rand() / RAND_MAX;
        }
    }
}

void initNetwork(int* layerSizes, int numLayers, Network* network) {
    network->numLatent = numLayers - 1;
    initLayer(layerSizes[0], 0, &network->input);
    network->latents = (Layer*) malloc(network->numLatent * sizeof(Layer));
    for(int i = 1; i < numLayers; i++) {
        initLayer(layerSizes[i], layerSizes[i - 1], &network->latents[i - 1]);
    }
}

void updateLayer(Layer* layer) {
    for(int i = 0; i < layer->numUpper; i++) {
        
    }
}