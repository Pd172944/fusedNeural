#ifndef QUANTIZED_H
#define QUANTIZED_H

#include "tensor.h"
#include <stdint.h>

typedef struct {
    int8_t *data;
    Shape shape;
    int num_elements;
    float scale;
    int32_t zero_point;
} QuantizedTensor;

QuantizedTensor* quantize_tensor(Tensor* float_tensor);
Tensor* dequantize_tensor(QuantizedTensor* qt);
void free_quantized_tensor(QuantizedTensor* qt);
QuantizedTensor* matmul_quantized(QuantizedTensor* a, QuantizedTensor* b);


#endif

