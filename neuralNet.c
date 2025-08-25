#include "neuralNet.h"
#include "tensor.h"
#include "operations.h"
#include <stdlib.h>


LinearLayer* create_linear_layer(int in_features, int out_features) {
    LinearLayer* layer = (LinearLayer*) malloc(sizeof(LinearLayer));

    int weight_dims[2] = {in_features, out_features};
    layer->weights = create_tensor(weight_dims, 2);
    set_random_data(layer->weights);

    int bias_dims[2] = {1, out_features};
    layer->biases = create_tensor(bias_dims, 2);
    set_random_data(layer->biases);

    return layer;
}

void free_linear_layer(LinearLayer* layer) {
    freeTensor(layer->weights);
    freeTensor(layer->biases);
    free(layer);
}

Tensor* linear_forward(LinearLayer* layer, Tensor* input) {
    //matmul
    Tensor* matmul_result = matmul_naive(input, layer->weights);
    //add biases
    Tensor* output_pre_relu = add(matmul_result, layer->biases);
    Tensor* output = create_tensor(output_pre_relu->shape.dims, output_pre_relu->shape.num_dims);

    for (int i = 0; i < output->num_elements; i++) {
        output->data[i] = (output_pre_relu->data[i] > 0) ? output_pre_relu->data[i] : 0;
    }

    output->backward_cache = output_pre_relu;
    
    //relu
    
    
    freeTensor(matmul_result); //clean intermediate tensor
    return output;
}


Tensor* linear_fused_forward(LinearLayer* layer, Tensor* input) {
    //alloc output tensor
    int out_dims[2] = {input->shape.dims[0], layer->weights->shape.dims[1]};
    Tensor* output = create_tensor(out_dims, 2);

    //fused op singular loop
    for (int i = 0; i < input->shape.dims[0]; i++) {
        for (int j = 0; j < layer->weights->shape.dims[1]; j++) {
            float sum = 0.0f;
            for (int k = 0; k < input->shape.dims[1]; k++) {
                sum += input->data[i * input->shape.dims[1] + k] * layer->weights->data[k * layer->weights->shape.dims[1] + j];
                
            }
            float val = sum + layer->biases->data[j]; //add bias and activate in same lop
            output->data[i * out_dims[1] + j] = (val > 0) ? val : 0; //relu
        }
    }
    return output;
}


// requires transpose and matmult on gradients
void linear_backward(Tensor* input, LinearLayer* layer, Tensor* output, Tensor* output_grad) {
    //backprop through ReLU
    Tensor* pre_relu_grad = create_tensor(output_grad->shape.dims, output_grad->shape.num_dims);
    Tensor* output_pre_relu = output->backward_cache; // Get pre-relu values
    for (int i = 0; i < output_grad->num_elements; i++) {
        pre_relu_grad->data[i] = (output_pre_relu->data[i] > 0) ? output_grad->data[i] : 0;
    }

    // backrpop through bias addition (gradient is simply passed)
    //Tensor* bias_grad = pre_relu_grad;

    // backprop through matrix multiplication
    Tensor* input_T = transpose(input); // transpose write
    Tensor* weight_grad = matmul_naive(input_T, pre_relu_grad);

    // 4. Update weights and biases
    for (int i = 0; i < layer->weights->num_elements; i++) {
        layer->weights->data[i] -= 0.01 * weight_grad->data[i]; // 0.01 is the learning rate
    }
    for (int i = 0; i < layer->biases->num_elements; i++) {
        layer->biases->data[i] -= 0.01 * pre_relu_grad->data[i];
    }
    
    freeTensor(pre_relu_grad);
    freeTensor(input_T);
    freeTensor(weight_grad);
    freeTensor(output_grad->backward_cache);
    freeTensor(output->backward_cache);
}


