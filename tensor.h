//headerfile for tensor

#ifndef TENSOR_H //if not def
#define TENSOR_H

typedef struct {
    int *dims; //dim arr ptr
    int num_dims; //num dims
} Shape;

typedef struct Tensor {
    float *data; //ptr to raw float data
    Shape shape; //tensor shape
    int num_elements; //dims[0]*dims[1]... total num
    struct Tensor* backward_cache;
} Tensor;


//function proto
Tensor* create_tensor(int* dims, int num_dims);
void freeTensor(Tensor* t);
void printTensor(Tensor* t);


#endif //TENSOR_H