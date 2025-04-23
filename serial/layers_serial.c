#include "layers_serial.h"

// Creates a convolutional layer.
// W: input Width, H: input height, D: input depth, K: filter width and height
// M: output feature maps, S: stride, P: zero padding
// Returns a pointer to the created convolutional layer.
Conv_Layer* make_conv_layer(int W, int H, int D, int K, int M, int S, int P) {
  // Allocate memory for the convolutional layer struct
  Conv_Layer* layer = malloc(sizeof(Conv_Layer));

  // Initialize layer parameters
  layer->in_width = W;
  layer->in_height = H;
  layer->in_depth = D;

  layer->filter_width = K;
  layer->num_filters = M;

  layer->stride = S;
  layer->padding = P;

  layer->out_width = (W - K + 2 * P) / S + 1;
  layer->out_height = (H - K + 2 * P) / S + 1;
  layer->out_depth = M;

  layer->out_size = layer->out_width * layer->out_height * layer->out_depth;

  // Allocate memory for weights and bias arrays
  layer->weights = malloc(sizeof(float) * K * K * M * D);
  layer->bias = malloc(sizeof(float) * M);

  return layer;
}

// Frees memory allocated for a convolutional layer.
// l: Pointer to the convolutional layer to be freed.
void free_conv(Conv_Layer* l) {
  free(l->bias);
  free(l->weights);
  free(l);
}

// Performs the forward pass for a convolutional layer.
// X: Input data, l: Convolutional layer, Y: Output data
void conv_forward(float* restrict X, Conv_Layer* l, float* restrict Y) {
  // For each output feature map
  for (int m = 0; m < l->out_depth; m++) {
    for (int j = 0; j < l->out_height; j++) {
      for (int i = 0; i < l->out_width; i++) {
        int y_idx = i + (l->out_width * (j + m * l->out_height)); // Output index
        // Calculate dot product of Weights*Input
        float sum = 0.0;
        for (int c = 0; c < l->in_depth; c++) {
          for (int f_j = 0; f_j < l->filter_width; f_j++) {
            for (int f_i = 0; f_i < l->filter_width; f_i++) {
              int f_idx = f_i + (f_j * l->filter_width) + (c + m * l->in_depth) * (l->filter_width * l->filter_width); // Filter Index
              int x_j = -l->padding + j * l->stride + f_j; // Input height index, increased by stride
              int x_i = -l->padding + i * l->stride + f_i; // Input width index, increased by stride
              // If in range of image, else zero
              if (x_j >= 0 && x_i >= 0 && x_j < l->in_height && x_i < l->in_width) {
                int x_idx = c * l->in_height * l->in_width + x_j * l->in_width + x_i; // Input index
                sum += l->weights[f_idx] * X[x_idx];
              } // if
            } // for f_i
          } // for f_j
        } // for c
        sum += l->bias[m]; // Add bias
        Y[y_idx] = sum; // Save output result
      } // for i
    } // for j
  } // for m
}

// Creates a ReLU activation layer.
// W: Input width, H: Input height, D: Input depth
// Returns a pointer to the created ReLU layer.
ReLU_Layer* make_relu_layer(int W, int H, int D) {
  // Allocate memory for the ReLU layer struct
  ReLU_Layer* layer = malloc(sizeof(ReLU_Layer));

  // Initialize layer parameters
  layer->in_width = W;
  layer->in_height = H;
  layer->in_depth = D;

  layer->out_width = W;
  layer->out_height = H;
  layer->out_depth = D;

  layer->out_size = layer->out_width * layer->out_height * layer->out_depth;

  return layer;
}

// Performs the forward pass for a ReLU activation layer.
// X: Input data, l: ReLU layer, Y: Output data
void relu_forward(float* restrict X, ReLU_Layer* l, float* restrict Y) {
  for (int i = 0; i < l->out_size; i++) {
    Y[i] = (X[i] < 0.0f) ? 0.0f : X[i];
  }
}

// Frees memory allocated for a ReLU activation layer.
// l: Pointer to the ReLU layer to be freed.
void free_relu(ReLU_Layer* l) {

  free(l);
}

// Creates a max pooling layer.
// W: Input width, H: Input height, D: Input depth
// K: Pooling width and height, S: Stride
// Returns a pointer to the created max pooling layer.
Pool_Layer* make_pool_layer(int W, int H, int D, int K, int S) {
  // Allocate memory for the max pooling layer struct
  Pool_Layer* layer = malloc(sizeof(Pool_Layer));

  // Initialize layer parameters
  layer->in_width = W;
  layer->in_height = H;
  layer->in_depth = D;

  layer->pool_width = K;
  layer->stride = S;

  layer->out_width = floor((W - K) / S + 1);
  layer->out_height = floor((H - K) / S + 1);
  layer->out_depth = D;

  layer->out_size = layer->out_width * layer->out_height * layer->out_depth;

  return layer;
}

// Performs the forward pass for a max pooling layer.
// X: Input data, l: Pooling layer, Y: Output data
void pool_forward(float* restrict X, Pool_Layer* l, float* restrict Y) {
  // For each output feature map
  for (int m = 0; m < l->out_depth; m++) {
    for (int j = 0; j < l->out_height; j++) {
      for (int i = 0; i < l->out_width; i++) {
        int y_idx = i + l->out_width * (j + m * l->out_height); // Output index
        // Find Max in pooling filter
        float max = -INFINITY;
        for (int p_j = 0; p_j < l->pool_width; p_j++) {
          for (int p_i = 0; p_i < l->pool_width; p_i++) {
            int x_j = j * l->stride + p_j; // Input height index, increased by stride
            int x_i = i * l->stride + p_i; // Input width index, increased by stride
            int x_idx = x_i + (x_j + m * l->in_height) * l->in_width; // Input index
            if (X[x_idx] > max) {
              max = X[x_idx];
            } // if max
          } // for p_i
        } // for p_j
        Y[y_idx] = max;
      } // for i
    } // for j
  } // for m
}

// Frees memory allocated for a max pooling layer.
// l: Pointer to the max pooling layer to be freed.
void free_pool(Pool_Layer* l) {
  free(l);
}

// Creates a fully connected layer.
// W: Input width, H: Input height, D: Input depth, num_neurons: Number of output neurons
// Returns a pointer to the created fully connected layer.
FC_Layer* make_fc_layer(int W, int H, int D, int num_neurons) {
  // Allocate memory for the fully connected layer struct
  FC_Layer* layer = malloc(sizeof(FC_Layer));

  // Initialize layer parameters
  layer->in_neurons = W * H * D;
  layer->in_width = W;
  layer->in_height = H;
  layer->in_depth = D;

  layer->out_width = 1;
  layer->out_height = 1;
  layer->out_depth = num_neurons;

  layer->out_size = layer->out_width * layer->out_height * layer->out_depth;

  layer->weights = (float*)malloc(sizeof(float) * layer->in_neurons * layer->out_depth);
  layer->bias = (float*)malloc(sizeof(float) * layer->out_depth);

  return layer;
}

// Performs the forward pass for a fully connected layer.
// X: Input data, l: Fully connected layer, Y: Output data
void fc_forward(float* restrict X, FC_Layer* l, float* restrict Y) {
  // For every output neuron
  for (int i = 0; i < l->out_depth; i++) {
    // Calculate dot product of input and weights
    float sum = l->bias[i]; // add bias
    for (int j = 0; j < l->in_neurons; j++) {
      int w_idx = j + i * l->in_neurons; // Weight index
      sum += X[j] * l->weights[w_idx];
    }
    Y[i] = sum;
  }
}

// Frees memory allocated for a fully connected layer.
// l: Pointer to the fully connected layer to be freed.
void free_fc(FC_Layer* l) {
  free(l->bias);
  free(l->weights);
  free(l);
}

// Loads pre-trained weights and biases for a convolutional layer from a file.
// l: Pointer to the convolutional layer, file_name: Name of the file containing pre-trained weights and biases
// Returns 0 on success, 1 on failure.
int load_conv(Conv_Layer* l, char* file_name) {
  int filter_width, filter_height, depth, filters;

  FILE* fin = fopen(file_name, "r");
  if (fin == NULL) {
    printf("Error opening conv_layer file!\n");
    return 1;
  }

  // Read layer parameters from the file
  fscanf(fin, "%d %d %d %d", &filter_width, &filter_height, &depth, &filters);

  // Validate layer parameters
  assert(filter_width == l->filter_width);
  assert(filter_height == l->filter_width);
  assert(depth == l->in_depth);
  assert(filters == l->num_filters);

  double val;
  // Read weights from the file
  for (int f = 0; f < filters; f++) {
    for (int i = 0; i < filter_width; i++) {
      for (int j = 0; j < filter_width; j++) {
        for (int d = 0; d < depth; d++) {
          fscanf(fin, "%lf", &val);
          int idx = i + j * filter_width + (d + f * depth) * (filter_width * filter_width);
          l->weights[idx] = (float)val;
        }
      }
    }
  }

  // Read biases from the file
  for (int d = 0; d < filters; d++) {
    fscanf(fin, "%lf", &val);
    int idx = d;
    l->bias[idx] = (float)val;
  }

  fclose(fin);

  return 0;
}

// Loads pre-trained weights and biases for a fully connected layer from a file.
// l: Pointer to the fully connected layer, filename: Name of the file containing pre-trained weights and biases
// Returns 0 on success, 1 on failure.
int load_fc(FC_Layer* l, const char* filename) {

  FILE* fin = fopen(filename, "r");
  if (fin == NULL) {
    printf("Error opening fc_layer file!\n");
    return 1;
  }

  int num_inputs;
  int output_depth;
  // Read layer parameters from the file
  fscanf(fin, "%d %d", &num_inputs, &output_depth);

  // Check if the dimensions match the layer dimensions
  assert(output_depth == l->out_depth);
  assert(num_inputs == l->in_neurons);

  double val;
  // Read weights from the file
  for (int i = 0; i < l->out_depth; i++) {
    for (int h = 0; h < l->in_height; h++) {
      for (int w = 0; w < l->in_width; w++) {
        for (int d = 0; d < l->in_depth; d++) {
          // Compute weights index: weights are stored first by depth.
          // Convert it to match the row-major array.
          int idx = w + (h + d * l->in_height) * l->in_width + i * l->in_neurons;
          fscanf(fin, "%lf", &val);
          l->weights[idx] = (float)val;
        }
      }
    }
  }

  // Read biases from the file
  for (int i = 0; i < l->out_depth; i++) {
    fscanf(fin, "%lf", &val);
    l->bias[i] = (float)val;
  }

  fclose(fin);

  return 0;
}

// Creates a softmax layer.
// W: Input width, H: Input height, D: Input depth
// Returns a pointer to the created softmax layer.
Softmax_Layer* make_softmax_layer(int W, int H, int D) {
  // Allocate memory for the softmax layer struct
  Softmax_Layer* layer = malloc(sizeof(Softmax_Layer));

  // Initialize layer parameters
  layer->in_width = W;
  layer->in_height = H;
  layer->in_depth = D;

  layer->out_width = 1;
  layer->out_height = 1;
  layer->out_depth = H * W * D;

  layer->out_size = layer->out_width * layer->out_height * layer->out_depth;

  layer->likelihoods = (float*)malloc(sizeof(float) * layer->out_depth);

  return layer;
}

// Performs the forward pass for a softmax layer.
// X: Input data, l: Softmax layer, Y: Output data
void softmax_forward(float* restrict X, Softmax_Layer* l, float* restrict Y) {
  // Compute max activation
  float max = X[0];
  for (int i = 1; i < l->out_depth; i++) {
    if (X[i] > max) {
      max = X[i];
    }
  }

  // Compute exponentials and total
  float total = 0.0f;
  for (int i = 0; i < l->out_depth; i++) {
    float e = exp(X[i] - max);
    total += e;
    l->likelihoods[i] = e;
  }

  // Normalize and output to sum to one
  for (int i = 0; i < l->out_depth; i++) {
    Y[i] = l->likelihoods[i] / total;
  }
}

// Frees memory allocated for a softmax layer.
// l: Pointer to the softmax layer to be freed.
void free_softmax(Softmax_Layer* l) {

  free(l->likelihoods);
  free(l);
}
