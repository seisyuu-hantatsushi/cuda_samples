#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <cudnn_ops_infer.h>

int hadamard_product(void){
    cudnnStatus_t result;
    uint32_t i=0,j=0;
    //uint32_t randstate = 2463534242;
    // Initialize cuDNN
    cudnnHandle_t handle = NULL;
    cudnnCreate(&handle);

    uint32_t xDataSize = 6*5;
    uint32_t yDataSize = 6*5;
    uint32_t zDataSize = 6*5;

    float* xDataHost = malloc(xDataSize*sizeof(float));
    float* yDataHost = malloc(yDataSize*sizeof(float));
    float* zDataHost = malloc(zDataSize*sizeof(float));
    float* xDataDevice = NULL;
    float* yDataDevice = NULL;
    float* zDataDevice = NULL;

    // Create tensor descriptors
    cudnnTensorDescriptor_t xDesc = NULL, yDesc = NULL, zDesc = NULL;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&zDesc);

    {
	const int batchSize = 1;
	const int channels = 1;
	const int height = 6;
	const int width = 5;
	const int dims[4] = {batchSize, channels, height, width};
	const int strides[4] = {channels * height * width, height * width, width, 1};
	cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 4, dims, strides);
	for(i = 0; i<batchSize*channels*height*width; i++){
	    //xDataHost[i] = (float)(xorshitf32(&randstate))/(float)(UINT32_MAX/2);
	    xDataHost[i] = (float)i;
	}
    }

    {
	const int batchSize = 1;
	const int channels = 1;
	const int height = 6;
	const int width = 5;
	const int dims[4] = {batchSize, channels, height, width};
	const int strides[4] = {channels * height * width, height * width, width, 1};
	cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_FLOAT, 4, dims, strides);
	for(i = 0; i<batchSize*channels*height*width; i++){
	    //yDataHost[i] = (float)(xorshitf32(&randstate))/(float)(UINT32_MAX/2);
	    yDataHost[i] = (float)i;
	}
    }

    {
	const int batchSize = 1;
	const int channels = 1;
	const int height = 6;
	const int width = 5;
	const int dims[4] = {batchSize, channels, height, width};
	const int strides[4] = {channels * height * width, height * width, width, 1};
	cudnnSetTensorNdDescriptor(zDesc, CUDNN_DATA_FLOAT, 4, dims, strides);
    }

    cudaMalloc((void **)&xDataDevice, xDataSize*sizeof(float));
    cudaMalloc((void **)&yDataDevice, yDataSize*sizeof(float));
    cudaMalloc((void **)&zDataDevice, zDataSize*sizeof(float));

    cudaMemcpy(xDataDevice, xDataHost, xDataSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(yDataDevice, yDataHost, xDataSize*sizeof(float), cudaMemcpyHostToDevice);

    cudnnOpTensorDescriptor_t opTensorDesc;
    result = cudnnCreateOpTensorDescriptor(&opTensorDesc);
    if(result != CUDNN_STATUS_SUCCESS){
	fprintf(stderr, "unable to create Op Tensor: %d\n", result);
	goto error_exit;
    }
    cudnnSetOpTensorDescriptor(opTensorDesc,
			       CUDNN_OP_TENSOR_MUL,
			       CUDNN_DATA_FLOAT,
			       CUDNN_PROPAGATE_NAN);
    // MUL tensors
    const float alphaA = 1.0f;
    const float alphaB = 1.0f;
    const float beta = 1.0f;
    result = cudnnOpTensor(handle,
			   opTensorDesc,
			   &alphaA, xDesc, xDataDevice,
			   &alphaB, yDesc, yDataDevice,
			   &beta,   zDesc, zDataDevice);

    if(result != CUDNN_STATUS_SUCCESS){
	fprintf(stderr, "unable to mul Tensor: %d\n", result);
	goto error_exit;
    }

    cudnnDestroyOpTensorDescriptor(opTensorDesc);

    cudaMemcpy(zDataHost, zDataDevice, zDataSize*sizeof(float), cudaMemcpyDeviceToHost);

    for(i=0;i<6;i++){
	printf("[");
	for(j=0;j<5;j++){
	    printf("%f ",zDataHost[i*5+j]);
	}
	printf("]\n");
    }

    printf("end of hadamard_product\n");
 error_exit:


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

