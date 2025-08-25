#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "tensor.h"
Tensor* matmul_naive(Tensor* a, Tensor* b);
Tensor* add(Tensor* a, Tensor* b);
void relu(Tensor* t);
void set_random_data(Tensor* t);
Tensor* transpose(Tensor* t);

#endif 