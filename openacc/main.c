#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
// #include <time.h> 

#include "layers.h"
#include "malloc2D.h"
#include "timer.h"

#define NUM_IMAGES 50000  // Number of Input Data
#define NUM_CLASSES 10  // Number of Classes, CIFAR-10
#define IMAGE_PIXELS 3072 // Number of pixels of each image

#define MAX_TRAINING_DATA 50000 // Max number of training data
#define MAX_BATCH_DATA 10000    // Max number of samples per batch    
#define LINE_SIZE 3073  // Bytes of one line in the binary data files


// Folder for .bin files of cifar dataset on my system.
const char* DATA_FOLDER = "../../cifar-10-batches-bin";

// Loading N samples from CIFAR-10 Dataset to Image[N][PIXEL] and Label[N]
int load_data(float** image, int* label, int N) {

    //Find how many batches I need and how many extra samples
    int batches = (N / MAX_BATCH_DATA) + 1;
    int samples = N % MAX_BATCH_DATA;
    int n = 0;  //Image index

    // printf("Batches: %d, Samples: %d.\n",batches,samples);
    char file_name[1024];

    //Loading the whole batches. If we need less than 10.000 images, the condition (b<batches) ends the loop.
    for (int b = 1; b < batches; b++) {
        // printf("Loading input batch %d...\n", b);

        sprintf(file_name, "%s/data_batch_%d.bin", DATA_FOLDER, b);

        FILE* fbin = fopen(file_name, "rb");
        if (fbin == NULL) {
            printf("File not found. Error reading .bin files.\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < MAX_BATCH_DATA; i++) {
            uint8_t data[LINE_SIZE];

            size_t bytesRead = fread(data, 1, LINE_SIZE, fbin);
            assert(bytesRead == LINE_SIZE);

            label[n] = data[0];
            size_t data_i = 1;
            for (int j = 0; j < IMAGE_PIXELS && data_i < LINE_SIZE; j++) {
                image[n][j] = (float)data[data_i++] / 255.0 - 0.5;
            }
            n++;
        }
        assert((n % MAX_BATCH_DATA) == 0);
        fclose(fbin);
    }

    // Loading less than 10.000 images from a batch
    if (samples != 0) {
        // printf("Loading input batch %d...\n", batches);

        sprintf(file_name, "%s/data_batch_%d.bin", DATA_FOLDER, batches);

        FILE* fbin = fopen(file_name, "rb");

        if (fbin == NULL) {
            printf("File not found. Error reading .bin files.\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < samples; i++) {
            uint8_t data[LINE_SIZE];

            size_t bytesRead = fread(data, 1, LINE_SIZE, fbin);
            assert(bytesRead == LINE_SIZE);

            label[n] = data[0];
            size_t data_i = 1;
            for (int j = 0; j < IMAGE_PIXELS && data_i < LINE_SIZE; j++) {
                image[n][j] = (float)data[data_i++] / 255.0 - 0.5;
            }
            n++;
        }
        fclose(fbin);
    }assert(n == N);

    return 0;
}

// Test Function
void arr2txt(float* arr, int N, int M, char* file_name);
void arr2txt_2(float** arr, int N, int M, char* file_name);

int main() {
    // const char *label_names[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
    struct timespec t1,t3;
    double ttotal = 0, t2;
    double time_conv1 = 0, time_relu1 = 0, time_pool1 = 0;
    double time_conv2 = 0, time_relu2 = 0, time_pool2 = 0;
    double time_conv3 = 0, time_relu3 = 0, time_pool3 = 0;
    double time_fc = 0, time_softmax = 0;
    printf("Parallel (if) Code\n");
    printf("CNN for %d images\n", NUM_IMAGES);

    // Image labels 
    int labels[NUM_IMAGES];

    // Input: Image Data
    float** restrict input = malloc2D(NUM_IMAGES, IMAGE_PIXELS);
    cpu_timer_start(&t1);
    // Load Image Data 
    load_data(input, labels, NUM_IMAGES);
#pragma acc enter data copyin(input[0:NUM_IMAGES][0:IMAGE_PIXELS])

    t2 = cpu_timer_stop(t1);
    ttotal += t2;
    printf("Load Data time:%f seconds\n", t2);

    // Network Layers
    cpu_timer_start(&t1);
    Conv_Layer* L1 = make_conv_layer(N_in, N_in, C_in, K1, M1, S1, P1);
    ReLU_Layer* L2 = make_relu_layer(L1->out_width, L1->out_height, L1->out_depth);
    Pool_Layer* L3 = make_pool_layer(L2->out_width, L2->out_height, L2->out_depth, K3, S3);
    Conv_Layer* L4 = make_conv_layer(L3->out_width, L3->out_height, L3->out_depth, K4, M4, S4, P4);
    ReLU_Layer* L5 = make_relu_layer(L4->out_width, L4->out_height, L4->out_depth);
    Pool_Layer* L6 = make_pool_layer(L5->out_width, L5->out_height, L5->out_depth, K6, S6);
    Conv_Layer* L7 = make_conv_layer(L6->out_width, L6->out_height, L6->out_depth, K7, M7, S7, P7);
    ReLU_Layer* L8 = make_relu_layer(L7->out_width, L7->out_height, L7->out_depth);
    Pool_Layer* L9 = make_pool_layer(L8->out_width, L8->out_height, L8->out_depth, K9, S9);
    FC_Layer* L10 = make_fc_layer(L9->out_width, L9->out_height, L9->out_depth, M10);
    Softmax_Layer* L11 = make_softmax_layer(L10->out_width, L10->out_height, L10->out_depth);
    t2 = cpu_timer_stop(t1);
    ttotal += t2;

    printf("Create Network time:%f seconds\n", t2);

    //Loading Layers' parameters
    cpu_timer_start(&t1);
    load_conv(L1, "../weights/layer1_conv.txt");
    load_conv(L4, "../weights/layer4_conv.txt");
    load_conv(L7, "../weights/layer7_conv.txt");
    load_fc(L10, "../weights/layer10_fc.txt");
    t2 = cpu_timer_stop(t1);
    ttotal += t2;

    printf("Load Network Parameters time:%f seconds\n", t2);

    //Allocate Outputs
    cpu_timer_start(&t1);

    float* restrict O1 = malloc(sizeof(float) * L1->out_size);
    float* restrict O2 = malloc(sizeof(float) * L2->out_size);
    float* restrict O3 = malloc(sizeof(float) * L3->out_size);
    float* restrict O4 = malloc(sizeof(float) * L4->out_size);
    float* restrict O5 = malloc(sizeof(float) * L5->out_size);
    float* restrict O6 = malloc(sizeof(float) * L6->out_size);
    float* restrict O7 = malloc(sizeof(float) * L7->out_size);
    float* restrict O8 = malloc(sizeof(float) * L8->out_size);
    float* restrict O9 = malloc(sizeof(float) * L9->out_size);
    float* restrict O10 = malloc(sizeof(float) * L10->out_size);
    float** restrict O11 = malloc2D(NUM_IMAGES, L11->out_size);
     t2 = cpu_timer_stop(t1);
    ttotal += t2;

    printf("Create Ouputs time:%f seconds\n", t2);

#pragma acc enter data create(O1[0:L1->out_size],O2[0:L2->out_size],O3[0:L3->out_size])
#pragma acc enter data create(O4[0:L4->out_size],O5[0:L5->out_size],O6[0:L6->out_size])
#pragma acc enter data create(O7[0:L7->out_size],O8[0:L8->out_size],O9[0:L9->out_size])
#pragma acc enter data create(O10[0:L10->out_size],O11[0:NUM_IMAGES][0:L11->out_size])



    //Net Forward
    cpu_timer_start(&t1);
    for (int i = 0; i < NUM_IMAGES; i++) {

        cpu_timer_start(&t3);
        conv_forward(input[i], L1, O1);
        time_conv1 += cpu_timer_stop(t3);

        cpu_timer_start(&t3);
        relu_forward(O1, L2, O2);
        time_relu1 += cpu_timer_stop(t3);

        cpu_timer_start(&t3);
        pool_forward(O2, L3, O3);
        time_pool1 += cpu_timer_stop(t3);

        cpu_timer_start(&t3);
        conv_forward(O3, L4, O4);
        time_conv2 += cpu_timer_stop(t3);

        cpu_timer_start(&t3);
        relu_forward(O4, L5, O5);
        time_relu2 += cpu_timer_stop(t3);

        cpu_timer_start(&t3);
        pool_forward(O5, L6, O6);
        time_pool2 += cpu_timer_stop(t3);

        cpu_timer_start(&t3);
        conv_forward(O6, L7, O7);
        time_conv3 += cpu_timer_stop(t3);

        cpu_timer_start(&t3);
        relu_forward(O7, L8, O8);
        time_relu3 += cpu_timer_stop(t3);

        cpu_timer_start(&t3);
        pool_forward(O8, L9, O9);
        time_pool3 += cpu_timer_stop(t3);

        cpu_timer_start(&t3);
        fc_forward(O9, L10, O10);
        time_fc += cpu_timer_stop(t3);

        cpu_timer_start(&t3);
        softmax_forward(O10, L11, O11[i]);
        time_softmax += cpu_timer_stop(t3);
    }

#pragma acc update self(O11[0:NUM_IMAGES][0:L11->out_size])
    t2 = cpu_timer_stop(t1);
    ttotal += t2;
    
    printf("\n");
    printf("Net Forward total time: %f seconds\n", t2);

    printf("    Time for conv1: %f seconds\n", time_conv1);
    printf("    Time for relu1: %f seconds\n", time_relu1);
    printf("    Time for pool1: %f seconds\n", time_pool1);
    printf("    Time for conv2: %f seconds\n", time_conv2);
    printf("    Time for relu2: %f seconds\n", time_relu2);
    printf("    Time for pool2: %f seconds\n", time_pool2);
    printf("    Time for conv3: %f seconds\n", time_conv3);
    printf("    Time for relu3: %f seconds\n", time_relu3);
    printf("    Time for pool3: %f seconds\n", time_pool3);
    printf("    Time for fc: %f seconds\n", time_fc);
    printf("    Time for softmax: %f seconds\n", time_softmax);

    printf("\n");
    printf("  Conv: %f seconds\n",time_conv1+time_conv2+time_conv3);
    printf("  ReLU: %f seconds\n",time_relu1+time_relu2+time_relu3);
    printf("  Pool: %f seconds\n",time_pool1+time_pool2+time_pool3);
    printf("    FC: %f seconds\n", time_fc);
    printf("  Softmax: %f seconds\n", time_softmax);
    printf("\n");

    // Results
    cpu_timer_start(&t1);
    int predictions[NUM_IMAGES];
    for (int n = 0; n < NUM_IMAGES; n++) {
        int class_o = 0;
        float max_o = O11[n][0];
        for (int i = 1; i < L11->out_depth; i++) {
            if (max_o < O11[n][i]) {
                class_o = i;
                max_o = O11[n][i];
            }
        }
        predictions[n] = class_o;
    }

    //Network Accuracy
    int correct_label = 0;
    for (int n = 0; n < NUM_IMAGES; n++) {
        if (predictions[n] == labels[n]) {
            correct_label++;
        }
    }

    printf("Net Accuracy: %.2f %% \n", 100 * (float)correct_label / NUM_IMAGES);
    t2 = cpu_timer_stop(t1);
    ttotal += t2;

    printf("Net Accuracy time:%f seconds\n", t2);


    // Free memory
    cpu_timer_start(&t1);
#pragma acc exit data delete(O1[0:L1->out_size],O2[0:L2->out_size],O3[0:L3->out_size])
#pragma acc exit data delete(O4[0:L4->out_size],O5[0:L5->out_size],O6[0:L6->out_size])
#pragma acc exit data delete(O7[0:L7->out_size],O8[0:L8->out_size],O9[0:L9->out_size])
#pragma acc exit data delete(O10[0:L10->out_size],O11[0:NUM_IMAGES][0:L11->out_size])


    free(O11);
    free(O10);
    free(O9);
    free(O8);
    free(O7);
    free(O6);
    free(O5);
    free(O4);
    free(O3);
    free(O2);
    free(O1);

    free_softmax(L11);
    free_fc(L10);

    free_pool(L9);
    free_relu(L8);
    free_conv(L7);

    free_pool(L6);
    free_relu(L5);
    free_conv(L4);

    free_pool(L3);
    free_relu(L2);
    free_conv(L1);

#pragma acc exit data delete(input[0:NUM_IMAGES][0:IMAGE_PIXELS])
    free(input);
    t2 = cpu_timer_stop(t1);
    ttotal += t2;

    printf("Free memory time:%f seconds\n", t2);
    printf("Total time:%f seconds\n", ttotal);

    printf("END!\n");

    return 0;
}

//Print an array to text file
void arr2txt(float* arr, int N, int M, char* file_name) {
    FILE* file = fopen(file_name, "w");

    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fprintf(file, "%d,%d,%d\n", N, N, M);
    for (int k = 0; k < M; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                int idx = ((j * N) + i) + (N * N) * k;
                fprintf(file, "%f ", arr[idx]);
            }
            fprintf(file, "\n");
        }
    }

    fclose(file);
}

void arr2txt_2(float** arr, int N, int M, char* file_name) {
    FILE* file = fopen(file_name, "w");

    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fprintf(file, "%d,%d,%d\n", N, N, M);
    for (int n = 0; n < NUM_IMAGES; n+=1200){
    for (int k = 0; k < M; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                int idx = ((j * N) + i) + (N * N) * k;
                fprintf(file, "%f ", arr[n][idx]);
            }
            fprintf(file, "\n");
        }
    }}

    fclose(file);
}
