#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neuron.h"
#include "lodepng.h"
#include "time.h"

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
    initNetwork(layerSizes, 4, &network);
    randomizeNetworkLatents(&network);
    
    // for(int i = 0; i < 3; i++) {
    //     trainNetwork(&network, images, 1, 100, 100);
    //     printf("Gen %d Loss: %f\n", i * 100, getLoss(&network));
        
    //     generateOutput(&network, 4000);

    //     for(int i = 0; i < 32; i++) {
    //         for(int j = 0; j < 32; j++) {
    //             imageout[i * 32 + j] = network.layers[0].lower[i * 32 + j].x;
    //         }
    //     }
    //     char name[30];
    //     sprintf(name, "../obamasout/%03d.png", i);
    //     encode(name, imageout);
    // }

    NetworkGPU gnet;
    copyNetworkToGPU(&network, &gnet);
    for(int i = 0; i < 15; i++) {
        trainNetworkGPU(gnet, images, 1, 400, 100);
        copyNetworkFromGPU(gnet, &network);

        printf("Gen %d Loss: %f\n", i * 100, getLoss(&network));
        
        generateOutputGPU(gnet, 4000);
        copyNetworkFromGPU(gnet, &network);

        for(int i = 0; i < 32; i++) {
            for(int j = 0; j < 32; j++) {
                imageout[i * 32 + j] = network.layers[0].lower[i * 32 + j].x;
            }
        }
        char name[30];
        sprintf(name, "../obamasout/%03d.png", i);
        encode(name, imageout);
    }

    float tcpu;
    float tgpu;
    clock_t start;
    clock_t end;

    start = clock();
    generateOutput(&network, 4000);
    end = clock();
    tcpu = (float) (end - start) * 1000 / (float)CLOCKS_PER_SEC;

    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++) {
            imageout[i * 32 + j] = network.layers[0].lower[i * 32 + j].x;
        }
    }
    encode("../obamasout/final.png", imageout);

    randomizeNetworkLatents(&network);
    
    copyNetworkToGPU(&network, &gnet);
    generateOutputGPU(gnet, 4000);
    copyNetworkFromGPU(gnet, &network);
    
    start = clock();
    for(int i = 0; i < 10; i++) {
        copyNetworkToGPU(&network, &gnet);
        generateOutputGPU(gnet, 4000);
        copyNetworkFromGPU(gnet, &network);
    }
    end = clock();
    tgpu = (float) (end - start) * 1000 / (float)CLOCKS_PER_SEC / 10;

    printf("CPU: %.3f, GPU: %.3f, Speedup: %.3f\n", tcpu, tgpu, tcpu/tgpu);

    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++) {
            imageout[i * 32 + j] = network.layers[0].lower[i * 32 + j].x;
        }
    }
    encode("../obamasout/finalgpu.png", imageout);

    for(int i = 0; i < 16; i++) {
        free(images[i]);
    }
    free(imageout);

}
