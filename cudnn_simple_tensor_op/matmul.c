

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cudnn.h>
#include <cudnn_backend.h>
#include <cudnn_ops_infer.h>

struct Tensor {
    uint32_t column;
    uint32_t row;
    cudnnBackendDescriptor_t desc;
    cudnnDataType_t dataType;
    float *pHost;
    void  *pDevice;
    size_t dataSize;
};

cudnnStatus_t createTensor(uint32_t column, uint32_t row, struct Tensor **ppTensor){
    cudnnStatus_t result;
    struct Tensor *pNewTensor = malloc(sizeof(struct Tensor));
    size_t requestSize;
    int64_t dims[3]    = {1, column, row};
    int64_t strides[3] = {column*row, row, 1};
    int64_t id;
    int64_t alignment = 4;

    if(pNewTensor == NULL){
	return CUDNN_STATUS_ALLOC_FAILED;
    }

    memset((void *)pNewTensor, 0x00, sizeof(*pNewTensor));

    pNewTensor->column   = column;
    pNewTensor->row      = row;
    pNewTensor->dataType = CUDNN_DATA_FLOAT;

    requestSize = (pNewTensor->column*pNewTensor->row)*sizeof(float);
    pNewTensor->pHost = malloc(requestSize);
    result = cudaMalloc(&pNewTensor->pDevice, requestSize);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &pNewTensor->desc);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendSetAttribute(pNewTensor->desc,
				      CUDNN_ATTR_TENSOR_DATA_TYPE,
				      CUDNN_TYPE_DATA_TYPE,
				      1,
				      &pNewTensor->dataType);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendSetAttribute(pNewTensor->desc,
				      CUDNN_ATTR_TENSOR_DIMENSIONS,
				      CUDNN_TYPE_INT64,
				      3,
				      &dims);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendSetAttribute(pNewTensor->desc,
				      CUDNN_ATTR_TENSOR_STRIDES,
				      CUDNN_TYPE_INT64,
				      3,
				      &strides);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    id = (int64_t)(pNewTensor);
    result = cudnnBackendSetAttribute(pNewTensor->desc,
				      CUDNN_ATTR_TENSOR_UNIQUE_ID,
				      CUDNN_TYPE_INT64,
				      1,
				      &id);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendSetAttribute(pNewTensor->desc,
				      CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
				      CUDNN_TYPE_INT64,
				      1,
				      &alignment);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendFinalize(pNewTensor->desc);
    if(result != CUDNN_STATUS_SUCCESS){
	fprintf(stderr, "unable to finalized tensor. %d\n", result);
	goto error_exit;
    }

    *ppTensor = pNewTensor;

    return CUDNN_STATUS_SUCCESS;

 error_exit:
    if(pNewTensor != NULL){

	if(pNewTensor->desc != NULL){
	    cudnnBackendDestroyDescriptor(pNewTensor->desc);
	}

	if(pNewTensor->pDevice != NULL){
	    cudaFree(pNewTensor->pDevice);
	}

	if(pNewTensor->pHost != NULL){
	    free(pNewTensor->pHost);
	}

	free(pNewTensor);
    }
    return result;
}

void destroyTensor(struct Tensor *pTensor){
    if(pTensor != NULL){

	if(pTensor->desc != NULL){
	    cudnnBackendDestroyDescriptor(pTensor->desc);
	}

	if(pTensor->pDevice != NULL){
	    cudaFree(pTensor->pDevice);
	}

	if(pTensor->pHost != NULL){
	    free(pTensor->pHost);
	}

	free(pTensor);
    }
}

struct Operator {
    cudnnBackendDescriptor_t desc;
    cudnnBackendDescriptor_t nodeDesc;
    cudnnDataType_t dataType;
    size_t numOfIns;
    size_t numOfOuts;
    struct Tensor **pIns;
    struct Tensor **pOuts;
};

cudnnStatus_t makeMutMal(struct Tensor *pInX, struct Tensor *pInY, struct Tensor *pOutZ, struct Operator **ppOperator){
    cudnnStatus_t result;
    struct Operator *pNewOperator = NULL;

    pNewOperator = malloc(sizeof(struct Operator));
    if(pNewOperator == NULL){
	result = CUDNN_STATUS_ALLOC_FAILED;
	goto error_exit;
    }
    memset(pNewOperator, 0x00, sizeof(*pNewOperator));
    pNewOperator->dataType = CUDNN_DATA_FLOAT;

    pNewOperator->pIns = malloc(sizeof(struct Tensor*)*2);
    if(pNewOperator->pIns == NULL){
	result = CUDNN_STATUS_ALLOC_FAILED;
	goto error_exit;
    }

    pNewOperator->pIns[0] = pInX;
    pNewOperator->pIns[1] = pInY;

    result = cudnnBackendCreateDescriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR,
					  &pNewOperator->desc);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendSetAttribute(pNewOperator->desc,
				      CUDNN_ATTR_MATMUL_COMP_TYPE,
				      CUDNN_TYPE_DATA_TYPE,
				      1,
				      &pNewOperator->dataType);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendFinalize(pNewOperator->desc);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
					  &pNewOperator->nodeDesc);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendSetAttribute(pNewOperator->nodeDesc,
				      CUDNN_ATTR_OPERATION_MATMUL_ADESC,
				      CUDNN_TYPE_BACKEND_DESCRIPTOR,
				      1,
				      &(pInX->desc));
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendSetAttribute(pNewOperator->nodeDesc,
				      CUDNN_ATTR_OPERATION_MATMUL_BDESC,
				      CUDNN_TYPE_BACKEND_DESCRIPTOR,
				      1,
				      &(pInY->desc));
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendSetAttribute(pNewOperator->nodeDesc,
				      CUDNN_ATTR_OPERATION_MATMUL_CDESC,
				      CUDNN_TYPE_BACKEND_DESCRIPTOR,
				      1,
				      &(pOutZ->desc));
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendSetAttribute(pNewOperator->nodeDesc,
				      CUDNN_ATTR_OPERATION_MATMUL_DESC,
				      CUDNN_TYPE_BACKEND_DESCRIPTOR,
				      1,
				      &(pNewOperator->desc));
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendFinalize(pNewOperator->nodeDesc);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    *ppOperator = pNewOperator;
    return CUDNN_STATUS_SUCCESS;

 error_exit:
    if(pNewOperator != NULL){
	if(pNewOperator->desc != NULL){
	    cudnnBackendDestroyDescriptor(pNewOperator->nodeDesc);
	}

	if(pNewOperator->desc != NULL){
	    cudnnBackendDestroyDescriptor(pNewOperator->desc);
	}

	if(pNewOperator->pIns != NULL){
	    free(pNewOperator->pIns);
	}

	if(pNewOperator->pOuts != NULL){
	    free(pNewOperator->pOuts);
	}

	free(pNewOperator);
    }

    return result;
}

void destroyOperator(struct Operator *pOperator){

    if(pOperator != NULL){
	if(pOperator->desc != NULL){
	    cudnnBackendDestroyDescriptor(pOperator->nodeDesc);
	}

	if(pOperator->desc != NULL){
	    cudnnBackendDestroyDescriptor(pOperator->desc);
	}

	if(pOperator->pIns != NULL){
	    free(pOperator->pIns);
	}

	if(pOperator->pOuts != NULL){
	    free(pOperator->pOuts);
	}

	free(pOperator);
    }

    return;
}

struct Graph {
    cudnnBackendDescriptor_t desc;
    size_t numOfOps;
    struct Operator **ppOps;
};

cudnnStatus_t makeGraph(cudnnHandle_t handle, size_t numOfOps, struct Operator **ppOps, struct Graph **ppGraph){
    cudnnStatus_t result;
    struct Graph *pNewGraph = malloc(sizeof(struct Graph));
    uint32_t i;

    //fprintf(stderr, "numOfOps = %u\n", numOfOps);
    if(pNewGraph == NULL){
	return CUDNN_STATUS_ALLOC_FAILED;
    }

    memset(pNewGraph, 0x00, sizeof(*pNewGraph));
    pNewGraph->ppOps = malloc(sizeof(struct Operator*)*numOfOps);
    if(pNewGraph->ppOps == NULL){
	result = CUDNN_STATUS_ALLOC_FAILED;
	goto error_exit;
    }

    for(i=0;i<numOfOps;i++){
	pNewGraph->ppOps[i] = ppOps[i];
    }

    result = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
					  &pNewGraph->desc);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    {
	cudnnBackendDescriptor_t *pOpDescs = malloc(sizeof(cudnnBackendDescriptor_t)*numOfOps);
	if(pOpDescs == NULL){
	    result = CUDNN_STATUS_ALLOC_FAILED;
	    goto error_exit;
	}
	for(i=0;i<numOfOps;i++){
	    pOpDescs[i] = pNewGraph->ppOps[i]->nodeDesc;
	}
	result = cudnnBackendSetAttribute(pNewGraph->desc,
					  CUDNN_ATTR_OPERATIONGRAPH_OPS,
					  CUDNN_TYPE_BACKEND_DESCRIPTOR,
					  numOfOps,
					  &pOpDescs[0]);
	if(result != CUDNN_STATUS_SUCCESS){
	    fprintf(stderr,"unable to set operator. %d\n", result);
	    free(pOpDescs);
	    goto error_exit;
	}
	free(pOpDescs);
    }

    result = cudnnBackendSetAttribute(pNewGraph->desc,
				      CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
				      CUDNN_TYPE_HANDLE,
				      1,
				      &handle);
    if(result != CUDNN_STATUS_SUCCESS){
	goto error_exit;
    }

    result = cudnnBackendFinalize(pNewGraph->desc);
    if(result != CUDNN_STATUS_SUCCESS){
	fprintf(stderr,"unable to finalize graph. %d\n",result);
	goto error_exit;
    }

    return CUDNN_STATUS_SUCCESS;

 error_exit:

    if(pNewGraph != NULL){
	if(pNewGraph->ppOps != NULL){
	    free(pNewGraph->ppOps);
	}

	if(pNewGraph->desc != NULL){
	    cudnnBackendDestroyDescriptor(pNewGraph->desc);
	}

	free(pNewGraph);
    }

    return result;
}

void destroyGraph(struct Graph *pGraph){
    if(pGraph != NULL){
	if(pGraph->ppOps != NULL){
	    free(pGraph->ppOps);
	}

	if(pGraph->desc != NULL){
	    cudnnBackendDestroyDescriptor(pGraph->desc);
	}

	free(pGraph);
    }
}

struct Heuristics {
    cudnnBackendDescriptor_t desc;
    
}

cudnnStatus_t make

void matmul(void){
    const uint32_t M=7,N=5,K=10;
    uint32_t i,j;
    cudnnStatus_t result;

    struct Tensor *xTensor = NULL, *yTensor = NULL, *zTensor = NULL;
    struct Operator *matmulOp = NULL;
    struct Graph *graph = NULL;

    // Initialize cuDNN
    cudnnHandle_t handle = NULL;
    cudnnCreate(&handle);

    result = createTensor(M, K, &xTensor);
    if(result != CUDNN_STATUS_SUCCESS){
	fprintf(stderr,"unable to create tensor. %d\n", result);
	goto error_exit;
    }

    result = createTensor(K, N, &yTensor);
    if(result != CUDNN_STATUS_SUCCESS){
	fprintf(stderr,"unable to create tensor. %d\n", result);
	goto error_exit;
    }

    result = createTensor(M, N, &zTensor);
    if(result != CUDNN_STATUS_SUCCESS){
	fprintf(stderr,"unable to create tensor. %d\n", result);
	goto error_exit;
    }

    for(i=0;i<M;i++){
	for(j=0;j<K;j++){
	    xTensor->pHost[i*K+j] = 1.0;
	}
    }

    for(i=0;i<K;i++){
	for(j=0;j<N;j++){
	    xTensor->pHost[i*N+j] = 1.0;
	}
    }

    for(i=0;i<M;i++){
	for(j=0;j<N;j++){
	    xTensor->pHost[i*N+j] = 0.0;
	}
    }

    cudaMemcpy(xTensor->pDevice, xTensor->pHost, xTensor->dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(yTensor->pDevice, yTensor->pHost, yTensor->dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(zTensor->pDevice, zTensor->pHost, yTensor->dataSize, cudaMemcpyHostToDevice);

    result = makeMutMal(xTensor, yTensor, zTensor, &matmulOp);
    if(result != CUDNN_STATUS_SUCCESS){
	fprintf(stderr, "unable to make matmul node. %d\n", result);
	goto error_exit;
    }

    {
	struct Operator *ops[] = { matmulOp };
	result = makeGraph(handle, 1, ops, &graph);
	if(result != CUDNN_STATUS_SUCCESS){
	    fprintf(stderr, "unable to make graph. %d\n", result);
	    goto error_exit;
	}
    }

    
 error_exit:

    destroyGraph(graph);
    
    destroyOperator(matmulOp);
    
    destroyTensor(xTensor);
    destroyTensor(yTensor);
    destroyTensor(zTensor);
    
    // Destroy cuDNN handle
    cudnnDestroy(handle);
}
