#ifndef NN_H
#define NN_H

#include "tensor.h"

typedef struct {
    Tensor* weights;
    Tensor* biases;
} LinearLayer;

LinearLayer* create_linear_layer(int in_features, int out_features);
void free_linear_layer(LinearLayer* layer);
Tensor* linear_forward(LinearLayer* layer, Tensor* input);

//fused forward
Tensor* linear_fused_forward(LinearLayer* layer, Tensor* input);
void linear_backward(Tensor* input, LinearLayer* layer, Tensor* output, Tensor* output_grad);

#endif