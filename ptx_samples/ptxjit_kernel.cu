
extern "C" __global__ void add(float *pX, float *pY, float *pZ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    pZ[tid] = pX[tid] + pY[tid];
}
