#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <limits.h>



int nextPowerOfTwo(int n) {
    if (n && !(n & (n - 1))) {
        return n;
    }

    int power = 1;
    while (power < n) {
        power <<= 1;
    }

    return power;
}
__global__ void BitonicSortHead(int* array, int i) {
    __shared__ int tempArray[512];
    int tx = threadIdx.x;
    int k = tx + blockDim.x * blockIdx.x;
    // ext means the target exchange idx, if k want to exchange with 5, then ext = 5
    // moved data from global memory to shared memoery in the SM, which is faster
    tempArray[tx] = array[k];
    __syncthreads();
    for (int j = i / 2; j > 0; j >>= 1) {
        int ext = k ^ j;
        // the exchange of A and B only need to check once, if we check A compare with B, we don't need to check B compare with A
        // we only check when the target is higher than current index
        // printf("k, k^j: %d, %d\n", k, ext);
        // printf("array[k], array[k^j]: %d, %d\n", array[k], array[ext]);
        if (ext > k && ext < blockDim.x * gridDim.x) {
            // now we want to compare, we know have to consider whether it's ascending or descending, based on the direction
            int dir = (k & i) == 0;
            // if dir == 0, that means ascending
            // k < ext but array[k] > array[ext] means it is descending, so i have to swap
            if (dir == (tempArray[tx] > tempArray[ext % blockDim.x])) {
                int t = tempArray[tx];
                tempArray[tx] = tempArray[ext % blockDim.x];
                tempArray[ext % blockDim.x] = t;
            }
        }
        __syncthreads();
    }
    array[k] = tempArray[tx];    
}


__global__ void BitonicSort(int* array, int j, int i) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    // ext means the target exchange idx, if k want to exchange with 5, then ext = 5
    int ext = k ^ j;
    // the exchange of A and B only need to check once, if we check A compare with B, we don't need to check B compare with A
    // we only check when the target is higher than current index
    // printf("k, k^j: %d, %d\n", k, ext);
    // printf("array[k], array[k^j]: %d, %d\n", array[k], array[ext]);
    if (ext > k) {
        // now we want to compare, we know have to consider whether it's ascending or descending, based on the direction
        int dir = (k & i) == 0;
        // if dir == 0, that means ascending
        // k < ext but array[k] > array[ext] means it is descending, so i have to swap
        if (dir == (array[k] > array[ext])) {
            int t = array[k];
            array[k] = array[ext];
            array[ext] = t;
        }
    }
}

__global__ void padFill(int* array, int n) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    if (k >= n) {
        array[k] = INT_MAX;
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    // ======================================================================
    // arCpu contains the input random array
    // arrSortedGpu should contain the sorted array copied from GPU to CPU
    // ======================================================================
    int* arrCpu = (int*)malloc(size * sizeof(int));
    int* arrSortedGpu = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // ======================================================================
    // Transfer data (arr_cpu) to device
    // ======================================================================

    // your code goes here .......
    int newSize;
    newSize = nextPowerOfTwo(size);
    // blocksize is number of thread
    int blockSize = 512;
    // gridsize is number of block
    int gridSize = (newSize + blockSize - 1) / blockSize;
    
    arrCpu = (int*)realloc(arrCpu, newSize * sizeof(int));
    arrSortedGpu = (int*)realloc(arrSortedGpu, newSize * sizeof(int));
    // for (int i = size; i < newSize; i++) {
    //     arrCpu[i] = 2147483647;
    // }
    
    int* arrGpu;
    cudaMalloc((void**) &arrGpu, newSize * sizeof(int));    
    cudaMemcpy(arrGpu, arrCpu, newSize * sizeof(int), cudaMemcpyHostToDevice);
    padFill<<<gridSize, blockSize>>>(arrGpu, size);
    // cudaMemcpy(arrCpu, arrGpu, newSize * sizeof(int), cudaMemcpyDeviceToHost);



    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);
    cudaEventRecord(start);
    
    // ======================================================================
    // Perform bitonic sort on GPU
    // ======================================================================

    // your code goes here .......
    



    // size_t freeMem, totalMem;
    // cudaMemGetInfo(&freeMem, &totalMem);
    // printf("Free memory: %zu, Total memory: %zu\n", freeMem, totalMem);

    
    // this outer loop is for bitonic merge
    for (int i = 2; i <= newSize; i <<= 1) {
        // second loop is for bitonic split
        if (i<= 512) {
            BitonicSortHead<<<gridSize, blockSize>>>(arrGpu, i);
        }
        else {
            for (int j = i / 2; j > 0; j >>= 1) {
            // printf("i, j: %d, %d\n", i, j);
            BitonicSort<<<gridSize, blockSize>>>(arrGpu, j, i);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                return 1;
            }
            cudaDeviceSynchronize();
        }
        }
    }
    

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

    // ======================================================================
    // Transfer sorted data back to host (copied to arr_sorted_gpu)
    // ======================================================================

    // your code goes here .......
    cudaMemcpy(arrSortedGpu, arrGpu, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(arrGpu);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;
    // printf("CPU\n");
    // for (int i = 0; i < size; i++) {
    //     printf("%d\n", arrCpu[i]);
    // }
    // printf("GPU\n");
    // for (int i = 0; i < size; i++) {
    //     printf("%d\n", arrSortedGpu[i]);
    // }

    int match = 1;
    for (int i = 0; i < size; i++) {
        // std::cout << i << std::endl;
        if (arrSortedGpu[i] != arrCpu[i]) {
            std::cout << i << " is wrong." << std::endl;
            match = 0;
            break;
        }
    }

    free(arrCpu);
    free(arrSortedGpu);

    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}

