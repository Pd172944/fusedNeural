#include "tensor.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


static int getNumElements(int* dims, int num_dims) {
    int count = 1;
    for (int i = 0; i < num_dims; i++) {
        count *= dims[i]; 
        //say dims is ptr to 1,2,3
        //3 dims
        //i = 0,1,2 do 1*2*3=6
    }
    return count;
}

//make new tensor
Tensor* create_tensor(int* dims, int num_dims) {
    Tensor* t = (Tensor*) malloc(sizeof(Tensor)); //size of tensor
    if (!t) {
        return NULL;
    }

    t->shape.num_dims = num_dims;
    t->shape.dims = (int*) malloc(num_dims * sizeof(int));

    if (!t->shape.dims) { //if some null t has a dims arr, free it 
        free(t);
        return NULL;
    }
    memcpy(t->shape.dims, dims, num_dims * sizeof(int));

    t->num_elements = getNumElements(dims, num_dims);
    t->data = (float*) malloc(t->num_elements * sizeof(float));
    if (!t->data) { //if some null t has data
        free(t->shape.dims);
        free(t);
        return NULL;
    }
    return t; 
}

void freeTensor(Tensor* t) {
    if (t) {
        free(t->shape.dims);
        free(t->data);
        free(t);
    }
}

//print 2d tensor
void printTensor(Tensor* t) {
    if (!t || t->shape.num_dims != 2) {
        printf("only printing 2d tensors for test");
        return;
    }

    printf("Tensor (Shape: %dx%d) :\n", t->shape.dims[0], t->shape.dims[1]);
    for (int i = 0; i < t->shape.dims[0]; i++) {
        for (int j = 0; j < t->shape.dims[1]; j++) {
            printf("%8.4f ", t->data[i * t->shape.dims[1] + j]);
        }
        printf("\n");
    }
}