#include "quantized.h"
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>


QuantizedTensor* quantize_tensor(Tensor* float_tensor) {
    float min_val = float_tensor->data[0];
    float max_val= float_tensor->data[0];

    for (int i = 1; i < float_tensor->num_elements; i++) {
        if (float_tensor->data[i] < min_val) min_val = float_tensor->data[i];
        if (float_tensor->data[i] > max_val) max_val = float_tensor->data[i];


    }

    QuantizedTensor* qt = (QuantizedTensor*) malloc(sizeof(QuantizedTensor));
    qt->shape.num_dims = float_tensor->shape.num_dims;
    qt->shape.dims = (int*)malloc(qt->shape.num_dims * sizeof(int));
    memcpy(qt->shape.dims, float_tensor->shape.dims, qt->shape.num_dims * sizeof(int));

    qt->num_elements = float_tensor->num_elements;
    qt->data = (int8_t*) malloc(qt->num_elements * sizeof(int8_t));

    qt->scale = (max_val - min_val) / (float) (SCHAR_MAX - SCHAR_MIN);
    qt->zero_point = SCHAR_MIN - round(min_val / qt->scale);

    for (int i = 0; i < qt->num_elements; i++) {
        qt->data[i] = round(float_tensor->data[i] / qt->scale) + qt->zero_point;
    }
    return qt;


}

void free_quantized_tensor(QuantizedTensor* qt) {
    if (qt) {
        free(qt->data);
        free(qt->shape.dims);
        free(qt);
    }
}

QuantizedTensor* matmul_quantized(QuantizedTensor* a, QuantizedTensor* b) {
    assert(a->shape.num_dims == 2 && b->shape.num_dims == 2);
    assert(a->shape.dims[1] == b->shape.dims[0]);

    int out_dims[2] = {a->shape.dims[0], b->shape.dims[1]};
    QuantizedTensor* result = (QuantizedTensor*) malloc(sizeof(QuantizedTensor));
    result->shape.num_dims = 2;
    result->shape.dims = (int*)malloc(2 * sizeof(int));
    result->shape.dims[0] = out_dims[0];
    result->shape.dims[1] = out_dims[1];
    result->num_elements = out_dims[0] * out_dims[1];
    result->data = (int8_t*) malloc(result->num_elements * sizeof(int8_t));

    // Calculate the output scale and zero point
    float out_scale = a->scale * b->scale;
    int32_t out_zero_point = 0; // Simplified for this example

    for (int i = 0; i < out_dims[0]; i++) {
        for (int j = 0; j < out_dims[1]; j++) {
            int32_t sum = 0; // Use a larger type for accumulation to prevent overflow
            for (int k = 0; k < a->shape.dims[1]; k++) {
                int32_t a_val = (int32_t)a->data[i * a->shape.dims[1] + k] - a->zero_point;
                int32_t b_val = (int32_t)b->data[k * b->shape.dims[1] + j] - b->zero_point;
                sum += a_val * b_val;
            }
            // Dequantize the final sum back to float
            float dequantized_sum = (float)sum * out_scale + out_zero_point;
            // Quantize to 8-bit integer
            result->data[i * out_dims[1] + j] = round(dequantized_sum / out_scale) + out_zero_point;
        }
    }
    return result;
}
    


