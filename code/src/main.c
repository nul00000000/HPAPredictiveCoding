#include "stdio.h"
#include "neuron.h"

int main() {
    Network network;
    int layerSizes[4] = {2, 3, 4, 1};
    float output[1] = {0};
    float input[2] = {0, 0};
    initNetwork(layerSizes, 4, &network);
    updateNetwork(&network);
    setNetworkInputs(&network, input);
    setNetworkOutputs(&network, output);
    printf("Loss: %f\n", getLoss(&network));
    for(int i = 0; i < 100; i++) {
        int a = rand() % 2;
        int b = rand() % 2;
        input[0] = (float) a;
        input[1] = (float) b;
        output[0] = (float) (a ^ b);
        for(int j = 0; j < 100; j++) {
            setNetworkInputs(&network, input);
            setNetworkOutputs(&network, output);
            updateNetworkInference(&network);
            updateNetwork(&network);
            // printNetwork(&network);
            // printf("%3d Loss: %f\n", j, getLoss(&network));
        }
        updateNetworkWeights(&network);
    }
    printf("Loss: %f\n", getLoss(&network));

    input[0] = 0;
    input[1] = 0;
    printf("Input: 00, ");
    evaluateNetwork(&network, input, output, 100);
    printf("Output: %f\n", output[0]);
    printf("Loss: %f\n", getLoss(&network));

    input[0] = 1;
    input[1] = 0;
    printf("Input: 01, ");
    evaluateNetwork(&network, input, output, 100);
    printf("Output: %f\n", output[0]);
    printf("Loss: %f\n", getLoss(&network));

    input[0] = 0;
    input[1] = 1;
    printf("Input: 10, ");
    evaluateNetwork(&network, input, output, 100);
    printf("Output: %f\n", output[0]);
    printf("Loss: %f\n", getLoss(&network));

    input[0] = 1;
    input[1] = 1;
    printf("Input: 11, ");
    evaluateNetwork(&network, input, output, 100);
    printf("Output: %f\n", output[0]);
    printf("Loss: %f\n", getLoss(&network));

}