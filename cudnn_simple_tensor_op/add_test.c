#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <cudnn_ops_infer.h>

int add_test(void){
    cudnnStatus_t result;
    uint32_t i;
    // Initialize cuDNN
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // Create tensor descriptors
    cudnnTensorDescriptor_t xDesc, yDesc, zDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&zDesc);

    // Set tensor descriptors
    const int batchSize = 1;
    const int channels = 3;
    const int height = 4;
    const int width = 4;
    const int dims[4] = {batchSize, channels, height, width};
    const int strides[4] = {channels * height * width, height * width, width, 1};
    cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 4, dims, strides);
    cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_FLOAT, 4, dims, strides);
    cudnnSetTensorNdDescriptor(zDesc, CUDNN_DATA_FLOAT, 4, dims, strides);

    // Allocate device memory for tensors
    const size_t dataSize = batchSize * channels * height * width * sizeof(float);

    float* xDataHost = malloc(dataSize);
    float* yDataHost = malloc(dataSize);
    float* zDataHost = malloc(dataSize);
    float* xDataDevice = NULL;
    float* yDataDevice = NULL;
    float* zDataDevice = NULL;

    cudaMalloc((void **)&xDataDevice, dataSize);
    cudaMalloc((void **)&yDataDevice, dataSize);
    cudaMalloc((void **)&zDataDevice, dataSize);

    for(i = 0; i< batchSize * channels * height * width; i++){
	xDataHost[i] = (float)i;
	yDataHost[i] = (float)2*i;
    }

    cudaMemcpy(xDataDevice, xDataHost, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(yDataDevice, yDataHost, dataSize, cudaMemcpyHostToDevice);

    cudnnOpTensorDescriptor_t opTensorDesc;
    result = cudnnCreateOpTensorDescriptor(&opTensorDesc);

    cudnnSetOpTensorDescriptor(opTensorDesc,
			       CUDNN_OP_TENSOR_ADD,
			       CUDNN_DATA_FLOAT,
			       CUDNN_PROPAGATE_NAN);
    // Add tensors
    const float alphaA = 1.0f;
    const float alphaB = 1.0f;
    const float beta = 1.0f;
    result = cudnnOpTensor(handle,
			   opTensorDesc,
			   &alphaA, xDesc, xDataDevice,
			   &alphaB, yDesc, yDataDevice,
			   &beta,   zDesc, zDataDevice);


    cudnnDestroyOpTensorDescriptor(opTensorDesc);

    cudaMemcpy(zDataHost, zDataDevice, dataSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < batchSize * channels * height * width; i++) {
	printf("zDataHost[%d]=%f\n", i, zDataHost[i]);
    }

    // Free device memory
    cudaFree(xDataDevice);
    cudaFree(yDataDevice);
    cudaFree(zDataDevice);

    free(xDataHost);
    free(yDataHost);
    free(zDataHost);
 
    // Destroy tensor descriptors
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(zDesc);

    // Destroy cuDNN handle
    cudnnDestroy(handle);
    return 0;
}
