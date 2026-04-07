#ifndef NEURON
#define NEURON

#define LR (0.01)

typedef struct neuron {
    float x; //latent value or input
    float err; //error
    float xn; //x next, prediction of next layer
    float a;
} Neuron;

typedef struct layer {
    int numUpper;
    int numLower;
    Neuron* upper;
    float* weights; //nullptr if input layer
} Layer;

typedef struct network {
    int numLatent;
    Layer input;
    Layer* latents;
} Network;

#endif