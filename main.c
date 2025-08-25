#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For time(NULL)
#include "tensor.h"
#include "neuralNet.h"
#include "loss.h"
#include "operations.h"
#include "quantized.h" // For QuantizedTensor and quantization functions

// A simple training function for demonstration
void train(LinearLayer* layer, Tensor* input, Tensor* target, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward Pass
        Tensor* output = linear_forward(layer, input);
        
        // Calculate Loss
        float loss = mse_loss(output, target);
        
        // Backpropagation: Get gradient of the loss with respect to the output
        Tensor* output_grad = mse_loss_gradient(output, target);
        
        // Backward Pass
        linear_backward(input, layer, output, output_grad);
        
        // Print progress
        if ((epoch + 1) % 50 == 0) {
            printf("Epoch %d: Loss = %f\n", epoch + 1, loss);
        }
        
        // Clean up temporary tensors from this loop iteration
        // output_pre_relu (backward_cache) is freed in linear_backward
        freeTensor(output);
        freeTensor(output_grad);
    }
}

void benchmark_forward_pass(LinearLayer* layer, Tensor* input) {
    printf("\n--- Benchmarking Naive Forward Pass ---\n");
    clock_t start_naive = clock();
    Tensor* output_naive = linear_forward(layer, input);
    clock_t end_naive = clock();
    double time_naive = (double)(end_naive - start_naive) / CLOCKS_PER_SEC;
    printf("Naive forward pass took: %f seconds\n", time_naive);
    freeTensor(output_naive); // Use free_tensor consistently

    printf("\n--- Benchmarking Fused Forward Pass ---\n");
    clock_t start_fused = clock();
    Tensor* output_fused = linear_fused_forward(layer, input); // FIX: Call fused forward
    clock_t end_fused = clock();
    double time_fused = (double)(end_fused - start_fused) / CLOCKS_PER_SEC;
    printf("Fused forward pass took: %f seconds\n", time_fused);
    freeTensor(output_fused); // Use free_tensor consistently

    printf("Speedup: %.2fx\n", time_naive / time_fused);
}

int main() {
    // Seed for random numbers
    srand(time(NULL));

    // Define synthetic data for a single training example
    int input_dims[2] = {1, 10};
    Tensor* input_data = create_tensor(input_dims, 2);
    set_random_data(input_data);

    int target_dims[2] = {1, 5};
    Tensor* target_data = create_tensor(target_dims, 2);
    set_random_data(target_data);

    // Create the linear layer
    LinearLayer* layer = create_linear_layer(10, 5);

    // Train the network
    train(layer, input_data, target_data, 500, 0.01);

    // Perform benchmarks
    benchmark_forward_pass(layer, input_data);

    // FIX: Quantization and its cleanup should happen BEFORE the main tensors are freed
    printf("\n--- Testing Quantized Model ---\n");
    QuantizedTensor* q_input = quantize_tensor(input_data);
    QuantizedTensor* q_weights = quantize_tensor(layer->weights);
    QuantizedTensor* q_output = matmul_quantized(q_input, q_weights);

    // You would typically dequantize and print/compare results here
    // For now, just printing a message:
    printf("Quantized matmul performed. Output is in q_output (int8_t).\n");

    // Clean up quantized tensors
    free_quantized_tensor(q_input);
    free_quantized_tensor(q_weights);
    free_quantized_tensor(q_output);

    // Final cleanup of original tensors
    freeTensor(input_data);
    freeTensor(target_data);
    free_linear_layer(layer);

    return 0;
}