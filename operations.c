#include "operations.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>


//naive matmul
Tensor* matmul_naive(Tensor* a, Tensor* b) {
    //shape validate
    assert(a->shape.num_dims == 2 && b->shape.num_dims == 2);
    assert(a->shape.dims[1] == b->shape.dims[0]);

    int result_dims[2] = {a->shape.dims[0], b->shape.dims[1]}; //r1, r2
    Tensor* result = create_tensor(result_dims, 2);
    
    for (int i = 0; i < result_dims[0]; i++) {
        for (int j = 0; j < result_dims[1]; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->shape.dims[1]; k++) {
                sum += a->data[i * a->shape.dims[1] + k] * b->data[k * b->shape.dims[1] + j];
            }
            result->data[i*result_dims[1] + j] = sum;
        }
    }
    return result;
}
    //add
    Tensor* add(Tensor* a, Tensor* b) {
        assert(a->num_elements == b->num_elements);

        Tensor* result = create_tensor(a->shape.dims, a->shape.num_dims);
        for (int i = 0; i < a->num_elements; i++) {
            result->data[i] = a->data[i]+b->data[i];
        }
        return result;
    }

    //ReLu activation this is zero out negative
    void relu(Tensor* t) {
        for (int i = 0; i < t->num_elements; i++) {
            if (t->data[i] < 0) {
                t->data[i] = 0;
            }
        }
    }

    //fill random data
    void set_random_data(Tensor* t) {
        for (int i = 0; i < t->num_elements; i++) {
            t->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; //rand between -1 and 1
        }
    }



Tensor* transpose(Tensor* t) {
    assert(t->shape.num_dims == 2);

    int new_dims[2] = {t->shape.dims[1], t->shape.dims[0]};
    Tensor* transposed_t = create_tensor(new_dims, 2);

    for (int i = 0; i < t->shape.dims[0]; i++) {
        for (int j = 0; j < t->shape.dims[1]; j++) {
            transposed_t->data[j * new_dims[1] + i] = t->data[i * t->shape.dims[1] + j];
        }
    }
    return transposed_t;



}