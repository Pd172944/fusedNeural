#include "loss.h"
#include <math.h>
#include <assert.h>

float mse_loss(Tensor* predictions, Tensor* targets) {
    assert(predictions->num_elements == targets->num_elements);

    float sum_sq_error = 0.0f;
    for (int i = 0; i < predictions->num_elements; i++) {
        float error = predictions->data[i] - targets->data[i];
        sum_sq_error += error*error;
    }
    return sum_sq_error / predictions->num_elements;
}

Tensor* mse_loss_gradient(Tensor* predictions, Tensor* targets) {
    Tensor* gradient = create_tensor(predictions->shape.dims, predictions->shape.num_dims);
    for (int i = 0; i < predictions->num_elements; i++) {
        gradient->data[i] = 2.0f * (predictions->data[i] - targets->data[i]) / predictions->num_elements;
    }
    return gradient;
}
