#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Constants defining the input data dimensions and parameters for each layer in the network.

#define N_in 32     // Input Width-Height
#define C_in 3      // Color Channels

// Layer 1: Convolutional
#define K1 5       // Filter Width Height
#define S1 1       // Stride
#define P1 2       // Padding
#define M1 16      // Output number of Feature Maps 

// Layer 2: ReLU (No parameters)

// Layer 3: Pooling
#define K3 2       // Pooling Width Height
#define S3 2       // Stride

// Layer 4: Convolutional
#define K4 5       // Filter Width Height
#define S4 1       // Stride
#define P4 2       // Padding
#define M4 20      // Output number of Feature Maps 

// Layer 5: ReLU (No parameters)

// Layer 6: Pooling
#define K6 2       // Pooling Width Height
#define S6 2       // Stride

// Layer 7: Convolutional
#define K7 5       // Filter Width Height
#define S7 1       // Stride
#define P7 2       // Padding
#define M7 20      // Output number of Feature Maps 

// Layer 8: ReLU (No parameters)

// Layer 9: Pooling
#define K9 2       // Pooling Width Height
#define S9 2       // Stride

// Layer 10: Fully Connected
#define M10 10     // Output Neurons

// Layer 11: Softmax (No parameters)

//-----------------------------------

// Structs representing different types of layers used in the Convolutional Neural Network (CNN).
// These structs contain the parameters and characteristics of each layer.

// In the network's data representation, three-dimensional data is stored in one-dimensional arrays
// using row-major order. For example, for an image pixel Image[d][h][w], the array index is 
// calculated as Image[w + h * Width + d * (Width * Height)].

// All layers include input and output width, height, and depth. The 'out_size' parameter denotes
// the output volume size, used for memory allocations throughout the program.

// Each struct is associated with the following functions:
//   - make_*layer(): Creates the layer, initializes variables and allocates arrays.
//   - free_*layer(): Deallocates memory used by the layer.
//   - *layer_forward(): Computes the forward pass of the layer.
//   - load_*layer(): Loads pre-trained data from files if the layer has weight parameters.


typedef struct conv_layer {
    int in_width;       // Input width
    int in_height;      // Input height
    int in_depth;       // Input depth

    int num_filters;    // Number of filters (output depth)
    int filter_width;   // Filter width
    int stride;         // Stride
    int padding;        // Padding

    int padded_height;  // Padding height
    int padded_width;   // Padding width
    int padded_size;    // Padding volume size
    
    int out_width;      // Output width
    int out_height;     // Output height
    int out_depth;      // Output depth

    int out_size;       // Output volume size
    int weights_size;   //Weights volume size

    float* weights;     // Filter weights
    float* bias;        // Bias values
    float* in_padded;   // Padded input

}Conv_Layer;

typedef struct relu_layer {
    int in_width;       // Input width
    int in_height;      // Input height
    int in_depth;       // Input depth

    int out_width;      // Output width
    int out_height;     // Output height
    int out_depth;      // Output depth

    int out_size;       // Output volume size

} ReLU_Layer;

typedef struct pool_layer {
    int in_width;       // Input width
    int in_height;      // Input height
    int in_depth;       // Input depth

    int pool_width;     // Pooling width
    int stride;         // Stride

    int out_width;      // Output width
    int out_height;     // Output height
    int out_depth;      // Output depth

    int out_size;       // Output volume size

} Pool_Layer;

typedef struct fc_layer {
    int in_width;       // Input width
    int in_height;      // Input height
    int in_depth;       // Input depth

    int in_neurons;     // Number of input neurons

    int out_width;      // Output width
    int out_height;     // Output height
    int out_depth;      // Output depth

    int out_size;       // Output volume size

    float* weights;     // Weight matrix
    float* bias;        // Bias values

}FC_Layer;

typedef struct softmax_layer {
    int in_width;       // Input width
    int in_height;      // Input height
    int in_depth;       // Input depth

    int out_width;      // Output width
    int out_height;     // Output height
    int out_depth;      // Output depth

    int out_size;       // Output volume size

    float* likelihoods; // Softmax probabilities

}Softmax_Layer;

// Functions to create layers
Conv_Layer* make_conv_layer(int W, int H, int D, int K, int M, int S, int P);
ReLU_Layer* make_relu_layer(int W, int H, int D);
Pool_Layer* make_pool_layer(int W, int H, int D, int K, int S);
FC_Layer* make_fc_layer(int W, int H, int D, int num_neurons);
Softmax_Layer* make_softmax_layer(int W, int H, int D);

// Functions to free memory allocated for layers
void free_conv(Conv_Layer* l);
void free_relu(ReLU_Layer* l);
void free_pool(Pool_Layer* l);
void free_fc(FC_Layer* l);
void free_softmax(Softmax_Layer* l);

// Functions for forward pass computation of layers
void conv_forward(float* restrict X, Conv_Layer* l, float* restrict Y);
void relu_forward(float* restrict X, ReLU_Layer* l, float* restrict Y);
void pool_forward(float* restrict X, Pool_Layer* l, float* restrict Y);
void fc_forward(float* restrict X, FC_Layer* l, float* restrict Y);
void softmax_forward(float* restrict X, Softmax_Layer* l, float* restrict Y);

// Functions to load pre-trained data for convolutional and fully connected layers
int load_conv(Conv_Layer* l, char* file_name);
int load_fc(FC_Layer* l, const char* filename);

#endif
