#include "stdio.h"
#include "neuron.h"
#include <stdlib.h>

//assumes 1-1-1-1 net
void printSimpleNet(Network* network) {
    printf("%.3f\n", network->layers[3].lower[0].x);
    printf(" |%.3f\n", network->layers[2].weights[0]);
    printf("%.3f %.3f %.3f %.3f\n", network->layers[2].lower[0].a, network->layers[2].lower[0].xh, network->layers[2].lower[0].x, network->layers[2].lower[0].err);
    printf("      |%.3f\n", network->layers[1].weights[0]);
    printf("     %.3f %.3f %.3f %.3f\n", network->layers[1].lower[0].a, network->layers[1].lower[0].xh, network->layers[1].lower[0].x, network->layers[1].lower[0].err);
    printf("           |%.3f\n", network->layers[0].weights[0]);
    printf("          %.3f %.3f %.3f %.3f\n", network->layers[0].lower[0].a, network->layers[0].lower[0].xh, network->layers[0].lower[0].x, network->layers[0].lower[0].err);
}

// int main() {
//     Network network;
//     int layerSizes[4] = {2, 2, 2, 2};
//     float output[2] = {0, 0};
//     float input[2] = {0, 0};
//     initNetwork(layerSizes, 4, &network);
//     randomizeNetworkLatents(&network);
//     updateNetwork(&network);
//     // printSimpleNet(&network);
//     float totalLoss = 0;
//     for(int i = 0; i < 100000; i++) {
//         float num = (float) rand() / RAND_MAX;
//         float num2 = (float) rand() / RAND_MAX;
//         input[0] = num;
//         input[1] = num2;
//         output[0] = num;
//         output[1] = num2;
//         randomizeNetworkLatents(&network);
//         for(int j = 0; j < 100; j++) {
//             setNetworkInputs(&network, input);
//             setNetworkOutputs(&network, output);
//             updateNetwork(&network);
//             updateNetworkInference(&network, 0);
//             if(i % 10000 == 0) {
//                 printf("Gen %d.%d, Loss: %f\n", i, j, getLoss(&network));
//             }
//         }
//         setNetworkInputs(&network, input);
//         setNetworkOutputs(&network, output);
//         updateNetwork(&network);
//         updateNetworkWeights(&network);
//         totalLoss += getLoss(&network);
//         if(i % 10000 == 0) {
//             printf("Gen %d, Loss: %f\n", i, totalLoss / 10000.0f);
//             totalLoss = 0;
//             // printSimpleNet(&network);
//         }
//     }
//     printf("Loss: %f\n", getLoss(&network));

//     // randomizeNetworkLatents(&network);

//     // for(int j = 0; j < 400; j++) {
//     //     updateNetwork(&network);
//     //     updateNetworkInference(&network, 1);
//     // }

//     // for(int i = 0; i < 100; i++) {
//     //     printf("%.3f ", network.layers[0].lower[i].x);
//     // }
//     // printf("\n");
// }

int main() {
    Network network;
    int layerSizes[4] = {100, 70, 30, 30};
    float output[1] = {0};
    float input[100];
    initNetwork(layerSizes, 4, &network);
    updateNetwork(&network);
    for(int j = 0; j < 100; j++) {
        input[j] = (float) j / 100.0f;
    }
    randomizeNetworkLatents(&network);
    for(int i = 0; i < 5000; i++) {
        // output[0] = (float) (a ^ b);
        // randomizeNetworkLatents(&network);
        for(int j = 0; j < 100; j++) {
            setNetworkInputs(&network, input);
            // setNetworkOutputs(&network, output);
            updateNetwork(&network);
            updateNetworkInference(&network, 0);
        }
        // printNetwork(&network);
        // printf("%3d Loss: %f\n", j, getLoss(&network));
        // if(i % 100 == 0)
        //     printf("Gen %d.%d Loss: %f\n", i, j, getLoss(&network));
        updateNetworkWeights(&network);
        // if(i % 10000 == 0)
            printf("Gen %d Loss: %f\n", i, getLoss(&network));
    }
    printf("Loss: %f\n", getLoss(&network));

    randomizeNetworkLatents(&network);

    for(int j = 0; j < 400; j++) {
        updateNetwork(&network);
        updateNetworkInference(&network, 1);
    }

    for(int i = 0; i < 100; i++) {
        printf("%.3f, ", network.layers[0].lower[i].x);
    }
    printf("\n");

    // input[0] = 0;
    // input[1] = 0;
    // printf("Input: 00, ");
    // evaluateNetwork(&network, input, output, 1000000);
    // printf("Output: %f\n", output[0]);
    // printf("Loss: %f\n", getLoss(&network));

    // input[0] = 1;
    // input[1] = 0;
    // printf("Input: 01, ");
    // evaluateNetwork(&network, input, output, 1000000);
    // printf("Output: %f\n", output[0]);
    // printf("Loss: %f\n", getLoss(&network));

    // input[0] = 0;
    // input[1] = 1;
    // printf("Input: 10, ");
    // evaluateNetwork(&network, input, output, 1000000);
    // printf("Output: %f\n", output[0]);
    // printf("Loss: %f\n", getLoss(&network));

    // input[0] = 1;
    // input[1] = 1;
    // printf("Input: 11, ");
    // evaluateNetwork(&network, input, output, 1000000);
    // printf("Output: %f\n", output[0]);
    // printf("Loss: %f\n", getLoss(&network));

}