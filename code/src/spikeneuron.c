#include "spikeneuron.h"

void initSpikeNeuron(SpikeNeuron* n) {
	n->x = 0.0f;
	n->spike = 0;
}

void initSpikeLayer(int numUpper, int numLower, SpikeLayer* layer) {
	layer->numUpper = numUpper;
	layer->numLower = numLower;
	layer->lower = (SpikeNeuron*) malloc(numLower * sizeof(SpikeNeuron));
	layer->error - (SpikeNeuron*) malloc(numLower * sizeof(SpikeNeuron));
	for(int i = 0; i < numLower; i++) {
		initSpikeNeuron(&layer->lower[i]);
		initSpikeNeuron(&layer->error[i]);
	}
	if(numUpper * numLower > 0) {
		layer->weights = (float*) malloc(numUpper * numLower * sizeof(float));
		for(int i = 0; i < numUpper * numLower; i++) {
			layer->weights[i] = (float) rand() / RAND_MAX;
		}
	} else {
		layer->weights = NULL;
	}
}

void initSpikeNetwork(int* layerSizes, int numLayers, SpikeNetwork* network) {
	network->numLayers = numLayers;
	network->layers = (SpikeLayer*) malloc(numLayers * sizeof(SpikeLayer));
	for(int i = 0; i < numLayers - 1; i++) {
		initSpikeLayer(layerSizes[i + 1], layerSizes[i], &network->layers[i]);
	}
	initSpikeLayer(0, layerSizes[numLayers - 1], &network->layers[numLayers - 1]);
}

void freeSpikeLayer(SpikeLayer* layer) {
	free(layer->lower);
	free(layer->error);
    if(layer->weights) free(layer->weights);
}

void freeSpikeNetwork(SpikeNetwork* network) {
    for(int i = 0; i < network->numLayers; i++) {
        freeSpikeLayer(&network->layers[i]);
    }
    free(network->layers);
}

//above: the layer above the one being processed
void processLayerSpikes(SpikeLayer* layer, SpikeLayer* above) {
    for(int i = 0; i < layer->numLower; i++) {
        layer->lower[i].x -= layer->error[i].spike * ERROR_WEIGHTS;
        layer->error[i].x += layer->lower[i].spike * ERROR_WEIGHTS;
        for(int j = 0; j < layer->numUpper; j++) {
            layer->lower[i].x += above->error[i].spike * layer->weights[i * layer->numUpper + j];
            above->error[i].x -= layer->lower[i].spike * layer->weights[i * layer->numUpper + j];
        }
    }
}

void processSpikes(SpikeNetwork* network) {
    for(int i = 0; i < network->numLayers - 1; i++) {
        processLayerSpikes(&network->layers[i], &network->layers[i + 1]);
    }
    processLayerSpikes(&network->layers[network->numLayers - 1], 0);
}

void createLayerSpikes(SpikeLayer* layer) {
    for(int i = 0; i < layer->numLower; i++) {
        layer->lower[i].spike = (layer->lower[i].x >= SPIKE_THRESHOLD);
        layer->error[i].spike = (layer->error[i].x >= SPIKE_THRESHOLD);
        if(layer->lower[i].spike) {
            layer->lower[i].x -= SPIKE_THRESHOLD;
        }
        if(layer->error[i].spike) {
            layer->error[i].x -= SPIKE_THRESHOLD;
        }
    }
}

void createSpikes(SpikeNetwork* network) {
    for(int i = 0; i < network->numLayers; i++) {
        createLayerSpikes(&network->layers[i]);
    }
}

void updateSpikeNetwork(SpikeNetwork* network) {
    processSpikes(network);
    createSpikes(network);
}
