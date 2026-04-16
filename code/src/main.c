#include <stdio.h>
#include <stdlib.h>
#include "neuron.h"
#include "lodepng.h"

//assumes 32x32
int decode(const char* filename, float* imagef) {
  int error;
  int width;
  int height;

  unsigned char* image = 0;

  error = lodepng_decode32_file(&image, &width, &height, filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  for(int i = 0; i < 32 * 32; i++) {
    int r = image[i * 4];
    int g = image[i * 4 + 1];
    int b = image[i * 4 + 2];
    imagef[i] = ((float) r + (float) g + (float) b) / 3.0f / 256.0f;
  }

  free(image);
  return error;
}

int encode(const char* filename, float* imagef) {
  int error;

  unsigned char* image = malloc(32 * 32 * 4);

  for(int i = 0; i < 32 * 32; i++) {
    int a = (int) (fmin(fmax(imagef[i], 0.0f), 1.0f) * 255.0f);
    image[i * 4] = (char) a;
    image[i * 4 + 1] = (char) a;
    image[i * 4 + 2] = (char) a;
    image[i * 4 + 3] = 255;
  }

  error = lodepng_encode32_file(filename, image, 32, 32);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

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

void print32x32Img(float* img) {
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++) {
            int ind = (int) (fmax(fmin(img[i * 32 + j], 1.0f), 0.0f) * 91.0f);
            printf("%c", cars[ind]);
            // printf("%d ", ind);
        }
        printf("\n");
    }
}

int main() {
    float* images[16];
    float* imageout = (float*) malloc(32 * 32 * sizeof(float));
    for(int i = 0; i < 16; i++) {
        images[i] = (float*) malloc(32 * 32 * sizeof(float));
        char name[30];
        sprintf(name, "../obamas/%da.png", i + 1);
        decode(name, images[i]);
        for(int j = 0; j < 32*32; j++) {
            imageout[j] += images[i][j] / 16.0f;
        }
    }

    encode("../obamasout/avg.png", imageout);

    Network network;
    int layerSizes[4] = {32*32, 100, 20, 5};
    float output[1] = {0};
    // float input[100];
    initNetwork(layerSizes, 4, &network);
    updateNetwork(&network);
    randomizeNetworkLatents(&network);
    for(int i = 0; i < 1000; i++) {
        // int c = rand() % 100;
        // c = rand() % 100;
        // for(int i = 0; i < 100; i++) {
        //     int di = c - i;
        //     if(di > 50) di -= 100;
        //     if(di <= -50) di += 100;
        //     input[i] = di * di * 0.0002f;
        // }
        int c = rand() % 16;
        randomizeNetworkLatents(&network);
        for(int j = 0; j < 100; j++) {
            setNetworkInputs(&network, images[c]);
            // setNetworkOutputs(&network, output);
            updateNetwork(&network);
            updateNetworkInference(&network, 0);
        }
        // printNetwork(&network);
        // printf("%3d Loss: %f\n", j, getLoss(&network));
        // if(i % 100 == 0)
        //     printf("Gen %d.%d Loss: %f\n", i, j, getLoss(&network));
        setNetworkInputs(&network, images[c]);
        // setNetworkOutputs(&network, output);
        updateNetwork(&network);
        updateNetworkWeights(&network);
        if(i % 100 == 0) {
            printf("Gen %d Loss: %f\n", i, getLoss(&network));
            randomizeNetworkLatents(&network);

            for(int j = 0; j < 4000; j++) {
                updateNetwork(&network);
                updateNetworkInference(&network, 1);
            }
            for(int i = 0; i < 32; i++) {
                for(int j = 0; j < 32; j++) {
                    imageout[i * 32 + j] = network.layers[0].lower[i * 32 + j].x;
                }
            }
            char name[30];
            sprintf(name, "../obamasout/%03d.png", i/100);
            encode(name, imageout);
        }
    }
    printf("Loss: %f\n", getLoss(&network));

    randomizeNetworkLatents(&network);

    for(int j = 0; j < 4000; j++) {
        updateNetwork(&network);
        updateNetworkInference(&network, 1);
    }

    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++) {
            imageout[i * 32 + j] = network.layers[0].lower[i * 32 + j].x;
        }
    }
    encode("../obamasout/final.png", imageout);

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

    for(int i = 0; i < 16; i++) {
        free(images[i]);
    }
    free(imageout);

}