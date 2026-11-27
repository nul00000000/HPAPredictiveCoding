#include <stdlib.h>

#define SPIKE_THRESHOLD (1.0f)
#define ERROR_WEIGHTS (1.0f)

typedef struct {
    float x;
    int spike;
} SpikeNeuron;

typedef struct {
    int numUpper;
    int numLower;
    SpikeNeuron* lower;
    SpikeNeuron* error;
    float* weights; //nullptr if top layer
} SpikeLayer;

typedef struct {
    int numLayers;
    SpikeLayer* layers;
} SpikeNetwork;

void initSpikeNeuron(SpikeNeuron* n);
void initSpikeLayer(int numUpper, int numLower, SpikeLayer* layer);
void initSpikeNetwork(int* layerSizes, int numLayers, SpikeNetwork* network);

void freeSpikeLayer(SpikeLayer* layer);
void freeSpikeNetwork(SpikeNetwork* network);

void processLayerSpikes(SpikeLayer* layer, SpikeLayer* above);
void processSpikes(SpikeNetwork* net);
void createLayerSpikes(SpikeLayer* layer);
void createSpikes(SpikeNetwork* net);

void updatePredictions(SpikeNetwork* net);
void updateInferenceCurrents(SpikeNetwork* net);

void updateSpikeNetwork(SpikeNetwork* net);

void trainSpikeNetwork(SpikeNetwork* net, float** inputs, int numSamples, int numTimesteps);
