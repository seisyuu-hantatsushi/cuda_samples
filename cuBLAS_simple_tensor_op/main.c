
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define M 3
#define K 2
#define N 3

static uint32_t xorshitf32(uint32_t *pState){
    uint32_t state = *pState;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    *pState = state;
    return *pState;
}

int main(int argc, char **argv){
    cudaError_t cudaError;
    cublasStatus_t stat;
    cublasLtHandle_t handle = NULL;
    cublasLtMatmulDesc_t opDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cublasOperation_t transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;
    float alpha = 1.0, beta = 1.0;
    float *srcA = NULL, *srcB = NULL, *srcC = NULL;
    float *columOrderA = NULL, *columOrderB = NULL, *columOrderC = NULL;
    float *deviceA = NULL, *deviceB = NULL, *deviceC= NULL;
    size_t workspaceSize = 4*1024*1024;
    void *workspace = NULL;
    uint32_t randstate = 2463534242;
    uint32_t i,j;
    int returnedResult;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    srcA = malloc(M*K*sizeof(float));
    if(srcA == NULL){
	goto error_exit;
    }

    srcB = malloc(K*N*sizeof(float));
    if(srcB == NULL){
	goto error_exit;
    }

    // column first order
    columOrderA = malloc(M*K*sizeof(float));
    if(columOrderA == NULL){
	goto error_exit;
    }

    // column first order
    columOrderB = malloc(K*N*sizeof(float));
    if(columOrderB == NULL){
	goto error_exit;
    }

    for(i=0;i<M;i++){
	for(j=0;j<K;j++){
	    srcA[i*K+j] = (float)(xorshitf32(&randstate))/(float)(UINT32_MAX);
	    //srcA[i*K+j] = 10*i+j;
	    columOrderA[j*M+i] = srcA[i*K+j];
	}
    }

    for(i=0;i<K;i++){
	for(j=0;j<N;j++){
	    srcB[i*N+j] = (float)(xorshitf32(&randstate))/(float)(UINT32_MAX);
	    //srcB[i*N+j] = 10*i+j;
	    columOrderB[j*K+i] = srcB[i*N+j];
	}
    }

    cudaError = cudaMalloc(&workspace, workspaceSize);
    if(cudaError != cudaSuccess){
	goto error_exit;
    }

    cudaError = cudaMalloc((void **)&deviceA, M*K*sizeof(float));
    if(cudaError != cudaSuccess){
	goto error_exit;
    }

    cudaError = cudaMalloc((void **)&deviceB, K*N*sizeof(float));
    if(cudaError != cudaSuccess){
	goto error_exit;
    }

    cudaError = cudaMalloc((void **)&deviceC, M*N*sizeof(float));
    if(cudaError != cudaSuccess){
	goto error_exit;
    }

    stat = cublasLtCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }

    stat = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }

    stat = cublasLtMatmulDescSetAttribute(opDesc,
					  CUBLASLT_MATMUL_DESC_TRANSA,
					  &transA,
					  sizeof(transA));
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }

    stat = cublasLtMatmulDescSetAttribute(opDesc,
					  CUBLASLT_MATMUL_DESC_TRANSB,
					  &transB,
					  sizeof(transB));
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }

    stat = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, M);
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }

    stat = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, K);
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }

    stat = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M);
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }

    cudaMemcpy(deviceA, columOrderA, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, columOrderB, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(deviceC, 0x00, M*N*sizeof(float));

    stat = cublasLtMatmulPreferenceCreate(&preference);
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }
    stat = cublasLtMatmulPreferenceSetAttribute(preference,
						CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
						&workspaceSize,
						sizeof(workspaceSize));
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }

    stat = cublasLtMatmulAlgoGetHeuristic(handle,
					  opDesc,
					  Adesc,
					  Bdesc,
					  Cdesc,
					  Cdesc,
					  preference,
					  1,
					  &heuristicResult,
					  &returnedResult);
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }

    if(returnedResult == 0){
	fprintf(stderr,"not supported");
	goto error_exit;
    }

    stat = cublasLtMatmul(handle,
			  opDesc,
			  &alpha,
			  deviceA,
			  Adesc,
			  deviceB,
			  Bdesc,
			  &beta,
			  deviceC,
			  Cdesc,
			  deviceC,
			  Cdesc,
			  &heuristicResult.algo,
			  workspace,
			  workspaceSize,
			  0);
    if(stat != CUBLAS_STATUS_SUCCESS){
	goto error_exit;
    }
    {
	float *pDeviceResult = malloc(M*N*sizeof(float));
	float *pHostResult = malloc(M*N*sizeof(float));
	printf("result\n");

	cudaMemcpy(pDeviceResult, deviceC, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	printf("srcA\n");
	for(i=0;i<M;i++){
	    printf("[");
	    for(j=0;j<K;j++){
		printf("%f ",srcA[i*K+j]);
	    }
	    printf("]\n");
	}

	printf("columnA\n");
	for(i=0;i<K;i++){
	    printf("[");
	    for(j=0;j<M;j++){
		printf("%f ",columOrderA[i*M+j]);
	    }
	    printf("]\n");
	}

	printf("srcB\n");
	for(i=0;i<K;i++){
	    printf("[");
	    for(j=0;j<N;j++){
		printf("%f ",srcB[i*N+j]);
	    }
	    printf("]\n");
	}

	for(i=0;i<N;i++){
	    for(j=0;j<M;j++){
		uint32_t k,l;
		uint32_t srcA_offset = i;
		uint32_t srcB_offset = j;
		pHostResult[i*M+j] = 0;
		for(k=0;k<K;k++){
		    pHostResult[i*M+j] += srcA[i*K+k]*srcB[j+N*k];
		}
		//printf("\n");
	    }
	}

	printf("deviceResult\n");
	for(i=0;i<M;i++){
	    printf("[");
	    for(j=0;j<N;j++){
		printf("%f ",pDeviceResult[i*N+j]);
	    }
	    printf("]\n");
	}

	printf("hostResult\n");
	for(i=0;i<M;i++){
	    printf("[");
	    for(j=0;j<N;j++){
		printf("%f ",pHostResult[i*N+j]);
	    }
	    printf("]\n");
	}

	if(pDeviceResult!= NULL){
	    free(pDeviceResult);
	}
	if(pHostResult != NULL){
	    free(pHostResult);
	}
    }
 error_exit:

    if(preference != NULL){
	cublasLtMatmulPreferenceDestroy(preference);
    }

    if(Adesc != NULL){
	cublasLtMatrixLayoutDestroy(Adesc);
    }

    if(Bdesc != NULL){
	cublasLtMatrixLayoutDestroy(Bdesc);
    }

    if(Cdesc != NULL){
	cublasLtMatrixLayoutDestroy(Cdesc);
    }

    if(opDesc != NULL){
	cublasLtMatmulDescDestroy(opDesc);
    }
    if(handle != NULL){
	cublasLtDestroy(handle);
    }

    if(deviceC != NULL){
	cudaFree(deviceC);
    }

    if(deviceB != NULL){
	cudaFree(deviceB);
    }

    if(deviceA != NULL){
	cudaFree(deviceA);
    }

    if(workspace != NULL){
	cudaFree(workspace);
    }

    if(columOrderA != NULL){
	free(columOrderA);
    }

    if(columOrderB != NULL){
	free(columOrderB);
    }

    if(srcB != NULL){
	free(srcB);
    }

    if(srcA != NULL){
	free(srcA);
    }
}
