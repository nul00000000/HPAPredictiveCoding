#include "neuron.h"

void initNeuron(Neuron* n) {
    n->a = 0;
    n->err = 0;
    n->xh = 0;
    n->x = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
}

void initLayer(int numUpper, int numLower, Layer* layer) {
    layer->numUpper = numUpper;
    layer->numLower = numLower;
    layer->lower = (Neuron*) malloc(numLower * sizeof(Neuron));
    for(int i = 0; i < numLower; i++) {
        initNeuron(&layer->lower[i]);
    }
    if(numUpper * numLower > 0) {
        layer->weights = (float*) malloc(numUpper * numLower * sizeof(float));
        //maybe randomize in another function?
        for(int i = 0; i < numUpper * numLower; i++) {
            layer->weights[i] = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

void initNetwork(int* layerSizes, int numLayers, Network* network) {
    network->numLayers = numLayers;
    network->layers = (Layer*) malloc(numLayers * sizeof(Layer));
    for(int i = 0; i < numLayers - 1; i++) {
        initLayer(layerSizes[i + 1], layerSizes[i], &network->layers[i]);
    }
    initLayer(0, layerSizes[numLayers - 1], &network->layers[numLayers - 1]);
}

float f(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float dfdx(float x) {
    float enx = exp(-x);
    return enx / ((1 + enx) * (1 + enx));
}

void updateLayer(Layer* layer, Layer* nextLayer) {
    for(int i = 0; i < layer->numLower; i++) {
        layer->lower[i].a = 0;
        for(int j = 0; j < layer->numUpper; j++) {
            layer->lower[i].a += nextLayer->lower[j].x * layer->weights[i * layer->numUpper + j];
        }
        layer->lower[i].xh = f(layer->lower[i].a);
        layer->lower[i].err = layer->lower[i].x - layer->lower[i].xh;
    }
}

void updateLayerInference(Layer* layer, Layer* prevLayer) {
    for(int i = 0; i < layer->numLower; i++) {
        float n = 0;
        for(int j = 0; j < prevLayer->numLower; j++) {
            //gain modulated error
            float h = dfdx(prevLayer->lower[j].a) * prevLayer->lower[j].err;
            n += prevLayer->weights[j * layer->numLower + i] * h;
        }
        layer->lower[i].x -= IR * (layer->lower[i].err - n);
    }
}

void updateLayerWeights(Layer *layer, Layer *nextLayer) {
    for(int i = 0; i < layer->numLower; i++) {
        for(int j = 0; j < layer->numUpper; j++) {
            //gain modulated error
            float h = dfdx(layer->lower[j].a) * layer->lower[j].err;
            layer->weights[i * layer->numUpper + j] += LR * h * nextLayer->lower[j].x;
        }
    }
}

void updateNetwork(Network* network) {
    for(int i = 0; i < network->numLayers - 1; i++) {
        updateLayer(&network->layers[i], &network->layers[i + 1]);
    }
}

void updateNetworkInference(Network* network) {
    for(int i = 1; i < network->numLayers; i++) {
        updateLayerInference(&network->layers[i], &network->layers[i - 1]);
    }
}

void updateNetworkWeights(Network* network) {
    for(int i = 0; i< network->numLayers - 1; i++) {
        updateLayerWeights(&network->layers[i], &network->layers[i + 1]);
    }
}

void setNetworkOutputs(Network* network, float* outputs) {
    Layer* top = &network->layers[network->numLayers - 1];
    for(int i = 0; i < top->numLower; i++) {
        top->lower[i].x = outputs[i];
    }
}

void setNetworkInputs(Network* network, float* inputs) {
    Layer* top = &network->layers[0];
    for(int i = 0; i < top->numLower; i++) {
        top->lower[i].x = inputs[i];
    }
}

//Should be run after updating the network
float getLoss(Network* network) {
    float loss = 0;
    for(int i = 0; i < network->numLayers; i++) {
        for(int j = 0; j < network->layers[i].numLower; j++) {
            loss += network->layers[i].lower[j].err * network->layers[i].lower[j].err;
        }
    }
    return loss;
}

void printLayer(Layer* layer) {
    for(int i = 0; i < layer->numLower; i++) {
        printf(" %2.3f ", layer->lower[i].x);
    }
    printf("\n");
}

void printNetwork(Network* network) {
    for(int i = 0; i < network->numLayers; i++) {
        printLayer(&network->layers[i]);
    }
}

void evaluateNetwork(Network* network, float* input, float* output, int numIters) {
    for(int i = 0; i < numIters; i++) {
        setNetworkInputs(network, input);
        updateNetworkInference(network);
        updateNetwork(network);
    }
    for(int i = 0; i < network->layers[network->numLayers - 1].numLower; i++) {
        output[i] = network->layers[network->numLayers - 1].lower[i].x;
    }
}