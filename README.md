# PCA-Demonstrate-Matrix-transposition-on-shared-memory
Comparing the Performance of the Rectangular Shared Memory Kernels with  grid (1,1) block (16,16)

## Aim:
To compare the performance of Rectangular Shared Memory Kernels with a grid (1,1) and a block (16,16) for Matrix Transposition.

## Procedure:
## Step 1:
Define constants for block dimensions, padding, and shared memory configurations.

## Step 2:
Implement a function to print data.

## Step 3:
Implement multiple kernel functions for performing matrix transposition using shared memory.

## Step 4:
Set up CUDA device, configure grid and block dimensions, allocate device memory, perform matrix transposition using each kernel function, copy results to host, optionally print the data, and free the allocated memory.

## Step 5:
Reset the CUDA device.

## Step 6:
Terminate the program.

## Program
```c
#include <stdio.h>
#include <cuda.h>

#define TILE_SIZE 2

// Matrix transposition kernel using shared memory
__global__ void transposeMatrix(int* input, int* output, int width, int height) {
    // Create shared memory tile
    __shared__ int tile[TILE_SIZE][TILE_SIZE];

    // Calculate global indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Load data from global memory to shared memory tile
    if (col < width && row < height) {
        int inputIndex = row * width + col;
        tile[threadIdx.y][threadIdx.x] = input[inputIndex];
    }

    // Synchronize threads to ensure all data is loaded to shared memory
    __syncthreads();

    // Calculate new indices for transposed matrix
    int transCol = blockIdx.y * blockDim.y + threadIdx.x;
    int transRow = blockIdx.x * blockDim.x + threadIdx.y;

    // Store transposed data to global memory
    if (transCol < height && transRow < width) {
        int outputIndex = transRow * height + transCol;
        output[outputIndex] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    int width = 4;
    int height = 4;
    int size = width * height;

    // Create input matrix
    int* input = (int*)malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++) {
        input[i] = i + 1;
    }

    for (int row = 0; row < width; row++) {
        for (int col = 0; col < height; col++) {
            printf("%d ", input[row * height + col]);
        }
        printf("\n");
    }

    int* d_input, * d_output;
    int* output = (int*)malloc(sizeof(int) * size);

    // Allocate memory on the device
    cudaMalloc(&d_input, sizeof(int) * size);
    cudaMalloc(&d_output, sizeof(int) * size);

    // Copy input matrix from host to device
    cudaMemcpy(d_input, input, sizeof(int) * size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Perform matrix transposition on the device
    transposeMatrix << <grid, block >> > (d_input, d_output, width, height);

    // Copy transposed matrix from device to host
    cudaMemcpy(output, d_output, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // Print the transposed matrix
    printf("Transposed Matrix:\n");
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < height; col++) {
            printf("%d ", output[row * height + col]);
        }
        printf("\n");
    }

    // Free memory on the device and host
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```

## Output:
![image](https://github.com/sherwin-roger0/PCA-Demonstrate-Matrix-transposition-on-shared-memory/assets/50732268/a96d412e-7f72-4e90-9cae-ae5dae8a23f5)![image](https://github.com/sherwin-roger0/PCA-Demonstrate-Matrix-transposition-on-shared-memory/assets/50732268/ba536f34-ad0f-42db-a779-0b46d67d8809)
## Result:
Thus,the program to compare the performance of Rectangular Shared Memory Kernels with a grid (1,1) and a block (16,16) for Matrix Transposition has been successfully executed.
