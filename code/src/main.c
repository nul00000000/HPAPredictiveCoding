#include "stdio.h"
#include "neuron.h"
#include "lodepng.h"
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

//assumes 128x128
int decode(const char* filename, float* imagef) {
  int error;
  int width;
  int height;

  unsigned char* image = 0;

  error = lodepng_decode32_file(&image, &width, &height, filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  for(int i = 0; i < 128 * 128; i++) {
    int r = image[i * 4];
    int g = image[i * 4 + 1];
    int b = image[i * 4 + 2];
    imagef[i] = ((float) r + (float) g + (float) b) / 3.0f / 256.0f;
  }

  free(image);
  return error;
}

char cars[92] = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";

void print128x128Img(float* img) {
    for(int i = 0; i < 128; i++) {
        for(int j = 0; j < 128; j++) {
            int ind = (int) (fmax(fmin(img[i * 128 + j], 1.0f), 0.0f) * 91.0f);
            printf("%c", cars[ind]);
            // printf("%d ", ind);
        }
        printf("\n");
    }
}

int main() {
    float* image = (float*) malloc(128 * 128 * sizeof(float));

    decode("../images/13.png", image);

    print128x128Img(image);

    Network network;
    int layerSizes[4] = {128*128, 128, 50, 10};
    float output[1] = {0};
    // float input[2] = {0, 0};
    initNetwork(layerSizes, 4, &network);
    updateNetwork(&network);
    randomizeNetworkLatents(&network);
    for(int i = 0; i < 10; i++) {
        // int a = rand() % 2;
        // int b = rand() % 2;
        // input[0] = (float) a;
        // input[1] = (float) b;
        // output[0] = (float) (a ^ b);
        randomizeNetworkLatents(&network);
        for(int j = 0; j < 100; j++) {
            setNetworkInputs(&network, image);
            // setNetworkOutputs(&network, output);
            updateNetwork(&network);
            updateNetworkInference(&network, 0);
        }
        // printNetwork(&network);
        // printf("%3d Loss: %f\n", j, getLoss(&network));
        // if(i % 100 == 0)
        //     printf("Gen %d.%d Loss: %f\n", i, j, getLoss(&network));
        setNetworkInputs(&network, image);
        // setNetworkOutputs(&network, output);
        updateNetwork(&network);
        updateNetworkWeights(&network);
        // if(i % 1000 == 0)
            printf("Gen %d Loss: %f\n", i, getLoss(&network));
    }
    printf("Loss: %f\n", getLoss(&network));

    // randomizeNetworkLatents(&network);

    // for(int j = 0; j < 400; j++) {
    //     updateNetwork(&network);
    //     updateNetworkInference(&network, 1);
    // }

    // for(int i = 0; i < 100; i++) {
    //     printf("%.3f, ", network.layers[0].lower[i].x);
    // }
    // printf("\n");

    // input[0] = 0;
    // input[1] = 0;
    // printf("Input: 00, ");
    // evaluateNetwork(&network, input, output, 100);
    // printf("Output: %f\n", output[0]);
    // printf("Loss: %f\n", getLoss(&network));

    // input[0] = 1;
    // input[1] = 0;
    // printf("Input: 01, ");
    // evaluateNetwork(&network, input, output, 100);
    // printf("Output: %f\n", output[0]);
    // printf("Loss: %f\n", getLoss(&network));

    // input[0] = 0;
    // input[1] = 1;
    // printf("Input: 10, ");
    // evaluateNetwork(&network, input, output, 100);
    // printf("Output: %f\n", output[0]);
    // printf("Loss: %f\n", getLoss(&network));

    // input[0] = 1;
    // input[1] = 1;
    // printf("Input: 11, ");
    // evaluateNetwork(&network, input, output, 100);
    // printf("Output: %f\n", output[0]);
    // printf("Loss: %f\n", getLoss(&network));

}