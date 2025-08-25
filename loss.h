#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

float mse_loss(Tensor* predictions, Tensor* targets);
Tensor* mse_loss_gradient(Tensor* predictions, Tensor* targets);

#endif
