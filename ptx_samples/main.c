#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

extern char _binary_obj_ptxjit_kernel_ptx_start[];
extern char _binary_obj_ptxjit_kernel_ptx_end[];

int main(int argc, char **argv){

    CUlinkState lState = 0;
    CUresult cuResult;
    CUdevice device;
    float walltime;
    char error_log[8192], info_log[8192];
    unsigned int logSize = sizeof(error_log);
    int drvVersion;
    int deviceCount;
    void *cuOut;
    size_t outSize;
    CUmodule hModule;
    CUfunction hFunction;
    CUcontext hCtx;

    _binary_obj_ptxjit_kernel_ptx_end[0] = '\0';
    printf("%s\n", _binary_obj_ptxjit_kernel_ptx_start);

    cuResult = cuInit(0);
    if(cuResult != CUDA_SUCCESS){
	goto error_exit;
    }

    cuResult = cuDeviceGetCount(&deviceCount);
    if(cuResult != CUDA_SUCCESS){
	goto error_exit;
    }
    printf("device Count = %d\n", deviceCount);

    cuResult = cuDeviceGet(&device, 0);
    if(cuResult != CUDA_SUCCESS){
	fprintf(stderr, "unable to get device. %d\n", cuResult);
	goto error_exit;
    }

    cudaDriverGetVersion(&drvVersion);
    printf("driver version: %d\n", drvVersion);

    cuResult = cuCtxGetCurrent(&hCtx);
    if(cuResult != CUDA_SUCCESS){
	fprintf(stderr, "unable to get context. %d\n", cuResult);
	goto error_exit;
    }

    cuResult = cuDevicePrimaryCtxRetain(&hCtx, device);
    if(cuResult != CUDA_SUCCESS){
	fprintf(stderr, "unable to get context. %d\n", cuResult);
	goto error_exit;
    }

    cuCtxPushCurrent(hCtx);
    {
	unsigned int flags = 0;
	cuCtxGetFlags(&flags);
	printf("ctx flags = %08x\n", flags);
    }
    {
	CUjit_option options[6] = {
	    CU_JIT_WALL_TIME,
	    CU_JIT_INFO_LOG_BUFFER,
	    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
	    CU_JIT_ERROR_LOG_BUFFER,
	    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
	    CU_JIT_LOG_VERBOSE
	};
	void *optionVals[6] = {
	    &walltime,
	    info_log,
	    (void *)(long)logSize,
	    error_log,
	    (void *)(long)logSize,
	    (void *)1
	};
	cuResult = cuLinkCreate(6, options, optionVals, &lState);
	if(cuResult != CUDA_SUCCESS){
	    fprintf(stderr, "unable to create linker. %d\n", cuResult);
	    goto error_exit;
	}
    }
    cuResult = cuLinkAddData(lState,
			     CU_JIT_INPUT_PTX,
			     _binary_obj_ptxjit_kernel_ptx_start,
			     strlen(_binary_obj_ptxjit_kernel_ptx_start)+1,
			     0, 0, 0, 0);
    if(cuResult != CUDA_SUCCESS){
	fprintf(stderr, "unable to add data to linker. %d\n", cuResult);
	goto error_exit;
    }

    cuResult = cuLinkComplete(lState, &cuOut, &outSize);
    if(cuResult != CUDA_SUCCESS){
	fprintf(stderr, "unable to complete linker. %d\n", cuResult);
	goto error_exit;
    }
    printf("link ptx\n");

    cuResult = cuModuleLoadData(&hModule, cuOut);
    if(cuResult != CUDA_SUCCESS){
	fprintf(stderr, "unable to load module. %d\n", cuResult);
	goto error_exit;
    }

    cuResult = cuModuleGetFunction(&hFunction, hModule, "add");
    if(cuResult != CUDA_SUCCESS){
	fprintf(stderr, "unable to get function entry. %d\n", cuResult);
	goto error_exit;
    }

    {
	uint32_t i;
	const uint32_t M=5,N=4;
	float *pX = malloc(M*N*sizeof(float));
	float *pY = malloc(M*N*sizeof(float));
	float *pZ = malloc(M*N*sizeof(float));

	for(i=0;i<M*N;i++){
	    pX[i] = 1.0;
	    pY[i] = 2.0;
	}
	void *pXDevice = NULL;
	void *pYDevice = NULL;
	void *pZDevice = NULL;
	cudaMalloc(&pXDevice, M*N*sizeof(float));
	cudaMalloc(&pYDevice, M*N*sizeof(float));
	cudaMalloc(&pZDevice, M*N*sizeof(float));

	cudaMemcpy(pXDevice, pX, M*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pYDevice, pY, M*N*sizeof(float), cudaMemcpyHostToDevice);

	void *args[3] = {&pXDevice, &pYDevice, &pZDevice};

	printf("launch\n");
	cuResult = cuLaunchKernel(hFunction, 64, 1, 1, M*N, 1, 1, 0, NULL, args, NULL);
	if(cuResult != CUDA_SUCCESS){
	    fprintf(stderr, "failed to launch kernel. %d\n", cuResult);
	    goto error_kernel;
	}

	printf("get result\n");
	cudaMemcpy(pZ, pZDevice, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	for(i=0;i<M;i++){
	    uint32_t j;
	    for(j=0;j<N;j++){
		printf("Z[%u][%u]=%f\n", i,j,pZ[i*N+j]);
	    }
	}

    error_kernel:
	cudaFree(pXDevice);
	cudaFree(pYDevice);
	cudaFree(pZDevice);
    }

 error_exit:
    cuModuleUnload(hModule);
    cuLinkDestroy(lState);

    return 0;
}
