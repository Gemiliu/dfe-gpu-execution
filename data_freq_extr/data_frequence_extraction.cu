#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>

#include "timer.h"
#include "check.h"

#ifndef _MSC_VER
#include <inttypes.h>
#else
typedef unsigned __int32 uint32_t;
#endif

/*
This program shows that for 100 data with type unsigned 32-bit, how to
quickly generate the table with their corresponding number of occurence.
*/
#define SIZE 100
#define MAX_SIZE_HASH_TABLE 128
int random_number = 1;
#define EPS 0.000001

template<typename T >
void setZeroValue(std::vector<T> &vec, T value) {
    for (unsigned int i = 0; i < vec.size(); ++i) {
        vec[i] = value;
    }
}

__device__ __host__ bool floatEquals(float a, float b) {
    if (fabs(a - b) < EPS) {
        return true;
    } else {
        return false;
    }
}

int random_berkovich(int* random_number)
{
  *random_number = (*random_number * 314159269 + 453806245) & ((1U << 31) - 1);
  return *random_number;
}

void separator()
{
  printf("====================================\n");
}

void display_occurrence_table(const float* hash_table, const int* occurrence_table, int table_size)
{
  int i;
  for (i = 0; i < table_size; i++)
  {
    if (occurrence_table[i] != 0)
    {
      printf("%10d %5d\n", (int)hash_table[i], occurrence_table[i]);
    }
  }
  separator();
}

void display_data_stream(const float* data_stream, int data_size)
{
  int number_of_data_per_line = 7;
  int i;
  for (i = 0; i < data_size; i++)
  {
    printf("%10d", (int)data_stream[i]);
    if (i % number_of_data_per_line == number_of_data_per_line - 1)
    {
      printf("\n");
    }
  }
  if (number_of_data_per_line % data_size != 0)
  {
    printf("\n");
  }
  separator();
}

int normal_hash(
    const float* data_stream,
    int data_size,
    int bits_on_tail_to_extract,
    float* hash_table,
    int* occurrence_table
) {
    int collision_times = 0;
    const uint32_t mask = (1 << bits_on_tail_to_extract) - 1;
    int temp_index;
    for (int i = 0; i < data_size; i++) {
        float element = data_stream[i];
        uint32_t elementUint = data_stream[i];
        temp_index = (int)(elementUint & mask);
        while (occurrence_table[temp_index] != 0 && !floatEquals(hash_table[temp_index], element)) {
          collision_times++;
          temp_index = (temp_index + 1) % (1 << bits_on_tail_to_extract);
        }
        hash_table[temp_index] = element;
        occurrence_table[temp_index]++;
    }
    return collision_times;
}

int clam_hash(
    const float* data_stream,
    int data_size,
    int bits_on_tail_to_extract,
    float* hash_table,
    int* occurrence_table
) {
    int collision_times = 0;
    const uint32_t mask = (1 << bits_on_tail_to_extract) - 1;
    int temp_index;

    for (int i = 0; i < data_size; i++) {
        float element = data_stream[i];
        uint32_t elementUint = data_stream[i];
        temp_index = (int)(elementUint & mask);
        temp_index += (temp_index >> 1);
        temp_index %= (1 << bits_on_tail_to_extract);
        while (occurrence_table[temp_index] != 0 && !floatEquals(hash_table[temp_index], element)) {
          collision_times++;
          temp_index = (temp_index + 1) % (1 << bits_on_tail_to_extract);
        }
        hash_table[temp_index] = element;
        occurrence_table[temp_index]++;
    }
    return collision_times;
}

double normal_distribution_CDF_approximation(double mean, double sigma) {
  // random number from 0 to 0.99999 with accuracy of 0.00001
  const int randomNumber = random_berkovich(&random_number);
  int modRandomNumber = (randomNumber % 100000);
  if (modRandomNumber == 0) {
    std::cout << "error - modRandomNumber equal 0. Fixed." << std::endl;
    modRandomNumber = 1;
  }
  double p = modRandomNumber / 100000.0;
  double z = log(p / (1 - p));
  double z1 = z * sigma + mean;
  if (z1 < 0) {
      std::cout << "error - z1 less then 0. Fixed." << std::endl;
      z1 *= -1;
  }
  return z1;
}

inline void cudaCheck(const cudaError_t &err, const std::string &mes) {
	if (err != cudaSuccess) {
		std::cout << (mes + " - " + cudaGetErrorString(err)) << std::endl;
		exit(EXIT_FAILURE);
	}
}

__global__ void setZeroValueKernel(float *vec, float value, int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        vec[index] = value;
    }
}

__global__ void setZeroValueKernel(int *vec, int value, int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        vec[index] = value;
    }
}

__global__ void normalHashKernel(
    const float* data_stream,
    int length,
    int bits_on_tail_to_extract,
    float *hash_table,
    int *occurrence_table,
    int *collision_times,
    const int sizeOfHashTable
) {
    const int index = blockIdx.x;
    const float *dataGl = &data_stream[index * length];
    float *hashTableGl = &hash_table[index * sizeOfHashTable];
    int *occurrenceTableGl = &occurrence_table[index * sizeOfHashTable];

    __shared__ float hashTable[MAX_SIZE_HASH_TABLE];
    __shared__ int occurrenceTable[MAX_SIZE_HASH_TABLE];
    __shared__ float data[SIZE];
    for (int i = threadIdx.x; i < sizeOfHashTable; ++i) {
        hashTable[i] = 0.0f;
        occurrenceTable[i] = 0;
    }
    for (int i = threadIdx.x; i < length; ++i) {
        data[i] = dataGl[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int collisionTimes = 0;
        const uint32_t mask = (1 << bits_on_tail_to_extract) - 1;
        int temp_index;
        for (int i = 0; i < length; i++) {
            const float element = data[i];
            const uint32_t elementUint = data[i];
            temp_index = (int)(elementUint & mask);
            while (occurrenceTable[temp_index] != 0 && !floatEquals(hashTable[temp_index], element)) {
              collisionTimes++;
              temp_index = (temp_index + 1) & mask;
            }
            hashTable[temp_index] = element;
            occurrenceTable[temp_index]++;
        }
        atomicAdd(collision_times, collisionTimes);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < sizeOfHashTable; ++i) {
        hashTableGl[i] = hashTable[i];
        occurrenceTableGl[i] = occurrenceTable[i];
    }
}

__global__ void clamHashKernel(
    const float* data_stream,
    int length,
    int bits_on_tail_to_extract,
    float *hash_table,
    int *occurrence_table,
    int *collision_times,
    const int sizeOfHashTable
) {
    const int index = blockIdx.x;
    const float *dataGl = &data_stream[index * length];
    float *hashTableGl = &hash_table[index * sizeOfHashTable];
    int *occurrenceTableGl = &occurrence_table[index * sizeOfHashTable];

    __shared__ float hashTable[MAX_SIZE_HASH_TABLE];
    __shared__ int occurrenceTable[MAX_SIZE_HASH_TABLE];
    __shared__ float data[SIZE];
    for (int i = threadIdx.x; i < sizeOfHashTable; ++i) {
        hashTable[i] = 0.0f;
        occurrenceTable[i] = 0;
    }
    for (int i = threadIdx.x; i < length; ++i) {
        data[i] = dataGl[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int collisionTimes = 0;
        const uint32_t mask = (1 << bits_on_tail_to_extract) - 1;
        int temp_index;
        for (int i = 0; i < length; i++) {
            const float element = data[i];
            const uint32_t elementUint = data[i];
            temp_index = (int)(elementUint & mask);
            temp_index += (temp_index >> 1);
            temp_index %= (1 << bits_on_tail_to_extract);
            while (occurrenceTable[temp_index] != 0 && !floatEquals(hashTable[temp_index], element)) {
              collisionTimes++;
              temp_index = (temp_index + 1) & mask;
            }
            hashTable[temp_index] = element;
            occurrenceTable[temp_index]++;
        }
        atomicAdd(collision_times, collisionTimes);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < sizeOfHashTable; ++i) {
        hashTableGl[i] = hashTable[i];
        occurrenceTableGl[i] = occurrenceTable[i];
    }
}

void testCUDAVersion(
    std::string hashName,
    std::vector<float> &hashTable,
    std::vector<int> &occurenceTable,
    int &collisionTimes,
    const std::vector<float> &data,
    const int countIter,
    const int length,
    const int countBlocks,
    const int bits_on_tail,
    const int sizeOfHashTable
) {
    Timer timer;
    cudaError_t err = cudaSuccess;
    if (length > (1 << bits_on_tail)) {
        throw std::string("Error: the size of the hash table is less than the number of data");
    }

    float *dataDev;
    const int sizeOfData = data.size();
    const int sizeOfDataBytes = sizeOfData * sizeof(float);
    err = cudaMalloc((void **)&dataDev, sizeOfDataBytes);
    cudaCheck(err, "failed to allocated dataDev");

    float *hashTableDev;
    const int sizeOfHashTableBytes = sizeOfHashTable * countBlocks * sizeof(float);
    err = cudaMalloc((void **)&hashTableDev, sizeOfHashTableBytes);
    cudaCheck(err, "failed to allocated hashTableDev");

    int *occurenceTableDev;
    const int sizeOfOccurenceTableBytes = sizeOfHashTable * countBlocks * sizeof(int);
    err = cudaMalloc((void **)&occurenceTableDev, sizeOfOccurenceTableBytes);
    cudaCheck(err, "failed to allocated occurenceTableDev");

    int *collisionTimesDev;
    err = cudaMalloc((void **)&collisionTimesDev, sizeof(int));
    cudaCheck(err, "failed to allocated collisionTimesDev");

    float computeTime = 0.0f;
    float computeTimeWithCopy = 0.0f;
    for (unsigned int iter = 0; iter < countIter; ++iter) {
        timer.begin("with copy");
        err = cudaMemcpy(dataDev, &data[0], sizeOfDataBytes, cudaMemcpyHostToDevice);
        cudaCheck(err, "failed to copy data to the GPU");
        timer.begin("compute");
        {
            int threadsPerBlock = 128;
            int blocksPerGrid = (countBlocks * sizeOfHashTable + threadsPerBlock - 1) / threadsPerBlock;
            setZeroValueKernel<<<blocksPerGrid, threadsPerBlock>>>(hashTableDev, 0.0f, sizeOfHashTable * countBlocks);
            setZeroValueKernel<<<blocksPerGrid, threadsPerBlock>>>(occurenceTableDev, 0, sizeOfHashTable * countBlocks);
            setZeroValueKernel<<<blocksPerGrid, threadsPerBlock>>>(collisionTimesDev, 0, 1);
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        cudaCheck(err, "failed to launch kernel - setZeroValueKernel");
        if (hashName.compare(std::string("norm")) == 0) {
            int threadsPerBlock = 32;
            int blocksPerGrid = countBlocks;

            normalHashKernel<<<blocksPerGrid, threadsPerBlock>>>(
                dataDev,
                length,
                bits_on_tail,
                hashTableDev,
                occurenceTableDev,
                collisionTimesDev,
                sizeOfHashTable
            );
        } else if (hashName.compare(std::string("clam")) == 0) {
            int threadsPerBlock = 32;
            int blocksPerGrid = countBlocks;

            clamHashKernel<<<blocksPerGrid, threadsPerBlock>>>(
                dataDev,
                length,
                bits_on_tail,
                hashTableDev,
                occurenceTableDev,
                collisionTimesDev,
                sizeOfHashTable
            );
        } else {
            throw std::string("a hashName is not valid.");
        }
        cudaDeviceSynchronize();
        timer.end("compute");
        err = cudaGetLastError();
        cudaCheck(err, "failed to launch kernel - hashKernel");

        err = cudaMemcpy(&hashTable[0], hashTableDev, sizeOfHashTableBytes, cudaMemcpyDeviceToHost);
        cudaCheck(err, "failed to copy hashTableDev to host");
        err = cudaMemcpy(&occurenceTable[0], occurenceTableDev, sizeOfHashTableBytes, cudaMemcpyDeviceToHost);
        cudaCheck(err, "failed to copy occurenceTableDev to host");
        err = cudaMemcpy(&collisionTimes, collisionTimesDev, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheck(err, "failed to copy collisionTimesDev to host");
        timer.end("with copy");
        // computeTimeGPU += timer.getTimeMillisecondsFloat("compute") + timer.getTimeMillisecondsFloat("change data");
        computeTime += timer.getTimeMillisecondsFloat("compute");
        computeTimeWithCopy += timer.getTimeMillisecondsFloat("with copy");
    }
    const int countOperations = countBlocks * length * 1;
    const float avgComputeTime = computeTime / countIter;
    const float avgComputeTimeWithCopy = computeTimeWithCopy / countIter;
    std::cout << "avg compute time GPU = " << avgComputeTime << " milliseconds" << std::endl;
    std::cout << "avg compute time(with copy) GPU = " << avgComputeTimeWithCopy << " milliseconds" << std::endl;
    std::cout << "Computational throughput GPU = " << countOperations / (avgComputeTime * 1e3) << " MB/s" << std::endl;
    std::cout << "Computational throughput(with copy) GPU = " << countOperations / (avgComputeTimeWithCopy * 1e3) << " MB/s" << std::endl;
    // display_occurrence_table(
    //     &hashTable[0],
    //     &occurenceTable[0],
    //     sizeOfHashTable * countBlocks
    // );

    err = cudaFree(dataDev);
    cudaCheck(err, "failed to free dataDev");
    err = cudaFree(hashTableDev);
    cudaCheck(err, "failed to free hashTableDev");
    err = cudaFree(occurenceTableDev);
    cudaCheck(err, "failed to free occurenceTableDev");
    err = cudaFree(collisionTimesDev);
    cudaCheck(err, "failed to free collisionTimesDev");
}

void testCPUVersion(
    std::string hashName,
    std::vector<float> &hashTable,
    std::vector<int> &occurenceTable,
    int &collisionTimes,
    const std::vector<float> &data,
    const int countIter,
    const int length,
    const int countBlocks,
    const int bits_on_tail,
    const int sizeOfHashTable
) {
    if (length > (1 << bits_on_tail)) {
        throw std::string("Error: the size of the hash table is less than the number of data");
    }
    Timer timer;
    float computeTime = 0.0f;
    for (unsigned int iter = 0; iter < countIter; ++iter) {
        timer.begin("compute");
        collisionTimes = 0;
        setZeroValue(hashTable, 0.0f);
        setZeroValue(occurenceTable, 0);
        if (hashName.compare(std::string("norm")) == 0) {
            for (unsigned int i = 0; i < countBlocks; ++i) {
                collisionTimes += normal_hash(
                    &data[i * length],
                    length,
                    bits_on_tail,
                    &hashTable[i * sizeOfHashTable],
                    &occurenceTable[i * sizeOfHashTable]
                );
            }
        } else if (hashName.compare(std::string("clam")) == 0) {
            for (unsigned int i = 0; i < countBlocks; ++i) {
                collisionTimes += clam_hash(
                    &data[i * length],
                    length,
                    bits_on_tail,
                    &hashTable[i * sizeOfHashTable],
                    &occurenceTable[i * sizeOfHashTable]
                );
            }
        } else {
            throw std::string("a hashName is not valid.");
        }
        timer.end("compute");
        computeTime += timer.getTimeMillisecondsFloat("compute");
    }
    const int countOperations = countBlocks * length * 1;
    const float avgComputeTime = computeTime / countIter;
    std::cout << "avg compute time CPU = " << avgComputeTime << " milliseconds" << std::endl;
    std::cout << "Computational throughput CPU = " << countOperations / (avgComputeTime * 1e3) << " MB/s" << std::endl;
    // display_occurrence_table(
    //     &hashTable[0],
    //     &occurenceTable[0],
    //     sizeOfHashTable * countBlocks
    // );
}

void testHash(
    const std::string &hashName,
    const std::vector<float> &data,
    const int countIter,
    const int length,
    const int countBlocks,
    const int bits_on_tail,
    const int sizeOfHashTable
) {
    std::cout << "testing " << hashName << " hash" << std::endl;
    int collisionTimesCpu;
    std::vector<float> hashTableCpu(sizeOfHashTable * countBlocks);
    std::vector<int> occurenceTableCpu(sizeOfHashTable * countBlocks);
    testCPUVersion(
        hashName,
        hashTableCpu,
        occurenceTableCpu,
        collisionTimesCpu,
        data,
        countIter,
        length,
        countBlocks,
        bits_on_tail,
        sizeOfHashTable
    );
    int collisionTimesCuda;
    std::vector<float> hashTableCuda(sizeOfHashTable * countBlocks);
    std::vector<int> occurenceTableCuda(sizeOfHashTable * countBlocks);
    testCUDAVersion(
        hashName,
        hashTableCuda,
        occurenceTableCuda,
        collisionTimesCuda,
        data,
        countIter,
        length,
        countBlocks,
        bits_on_tail,
        sizeOfHashTable
    );
    std::cout << "collisionTimesCpu - " << collisionTimesCpu << std::endl;
    std::cout << "collisionTimesCuda - " << collisionTimesCuda << std::endl;
    if (collisionTimesCuda != collisionTimesCpu) {
        throw std::string("The results of the cpu and the cuda version have difference value - collisionTimes.");
    }
    for (int i = 0; i < hashTableCuda.size(); ++i) {
        if (!floatEquals(hashTableCpu[i], hashTableCuda[i])) {
            throw std::string("The results of the cpu and the cuda version have difference value - hashTable.");
        }
    }
    for (int i = 0; i < occurenceTableCuda.size(); ++i) {
        if (!floatEquals(occurenceTableCpu[i], occurenceTableCuda[i])) {
            throw std::string("The results of the cpu and the cuda version have difference value - occurenceTable.");
        }
    }
    std::cout << "Great! the cpu and the cuda impl of " << hashName << " hash work identically." << std::endl;
}

int main(int argc, char * argv[]) {
	try {
	    int dataSource = 1;
	    int countBlocks = 10;
	    int countIter = 1000;
        const int length = SIZE;
        if (argc > 1) {
	        countBlocks = atoi(argv[1]);
	    }
	    if  (argc > 2) {
	        dataSource = atoi(argv[2]);
	    }
	    const int sizeOfData = length * countBlocks;
        std::vector<float> data(sizeOfData, 0.0f);
        if (dataSource == 1) {
            for (int i = 0; i < sizeOfData; ++i) {
                data[i] = random_berkovich(&random_number) % 1000;
            }
        } else if (dataSource == 2) {
            // double mean = 167;
            double mean = 350;
            double sigma = 23;
            for (int i = 0; i < sizeOfData; ++i) {
                data[i] = (uint32_t)normal_distribution_CDF_approximation(mean, sigma);
            }
        } else {
            throw std::string("invalid value for dataSource");
        }
        // display_data_stream(&data[0], data.size());
        const int bits_on_tail = 7;
        const int sizeOfHashTable = 1 << bits_on_tail;

        testHash(
            "norm",
            data,
            countIter,
            length,
            countBlocks,
            bits_on_tail,
            sizeOfHashTable
        );
        testHash(
            "clam",
            data,
            countIter,
            length,
            countBlocks,
            bits_on_tail,
            sizeOfHashTable
        );
    } catch (const std::string &mes) {
        std::cout << mes << std::endl;
    }
    // getchar();
    return 0;
}
