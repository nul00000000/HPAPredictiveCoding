typedef struct neuron {
    float x; //latent value or input
    float err; //error
    float xn; //x next, prediction of next layer
} Neuron;