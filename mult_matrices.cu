#include <bits/stdc++.h>
#include <cuda.h>
#define H 500
#define W 500

using namespace std;

void llenar(int* v) {
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      v[i*W+j] = rand() % 10;
    }
  }
}

//complexity O((H**2)*W)
void mult(int *A, int *B,int *C) {
  int sum;
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      sum = 0;
      for (int k = 0; k < H; ++k)
        sum += A[i*W+k]* B[k*W+j];
     C[i*W+j] = sum;
    }
  }
}

void mostrar(int *v) {
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      cout<<v[i*W+j]<<" ";
    }
    cout<<endl;
  }
}

//Kuda function
__global__ void mult_dist(int *d_A, int *d_B,int *d_C){
  int i = blockIdx.y*blockDim.y+threadIdx.y;//todos los valores columna
  int j = blockIdx.x*blockDim.x+threadIdx.x;//todos los valores fial
  if(i < H && j < W){
    int Pvalue = 0;
    for(int k=0; k<H; k++){
       Pvalue += d_A[i*W+k] * d_B[k*W+j];
    }
    d_C[i*W+j] = Pvalue;
  }
}

int main() {
  clock_t start, end;
  double cpu_time_used;

//multiplicacion normal
  int size = H*W*sizeof(int);//size de la "matriz" unidimensional
  int *A = (int*)malloc(size);
  int *B = (int*)malloc(size);
  int *C = (int*)malloc(size);//resultado Normal
  int *D = (int*)malloc(size);//resultado Kuda

  llenar(A);
  llenar(B);

  start = clock();

  mult(A,B,C);

  end = clock();

  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  cout<<"\nTiempo invertido CPU ="<<fixed<<setprecision(6)<<cpu_time_used<<" s\n\n";

  //cout<<"Primera matriz:\n";
  //mostrar(A);
  //cout<<"Segunda matriz:\n";
  //mostrar(B);
  //cout<<"Matriz resultante:\n";
  //mostrar(C);


//------------------------------------------------------------
//multiplicacion utilizando Kuda
  int *d_A,*d_B,*d_D;
  int blockSize;
  int minGridSize;
  int gridSize;

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mult_dist, 0, H*W);
  /*
  minGridSize     = Suggested min grid size to achieve a full machine launch.
  blockSize       = Suggested block size to achieve maximum occupancy.
  func            = Kernel function.
  dynamicSMemSize = Size of dynamically allocated shared memory. Of course,
    it is known at runtime before any kernel launch. The size of the statically allocated
    shared memory is not needed as it is inferred by the properties of func.
  blockSizeLimit  = Maximum size for each block. In the case of 1D kernels,
    it can coincide with the number of input elements.
  */

  //comes from cuda_runtime.h
  gridSize = ((H*W) + blockSize - 1) / blockSize;
  cout<<"GridSize: "<<gridSize<<"\nBlockSize: "<<blockSize<<endl;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_D, size);

  start = clock();
  cudaMemcpy(d_A, A, sizeof(int)*H*W, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(int)*H*W, cudaMemcpyHostToDevice);
  //add<<<(N + M-1) / M, M>>>(d_a, d_b, d_c, N);
  mult_dist<<<gridSize, blockSize>>>(d_A, d_B, d_D);

  cudaMemcpy(D,d_D,sizeof(int)*H*W,cudaMemcpyDeviceToHost);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  cout<<"Tiempo invertido GPU ="<<fixed<<setprecision(6)<<cpu_time_used<<" s\n\n";

  //cout<<"Matriz resultante con Kuda:\n";
  //mostrar(D);
  free(A); free(B); free(C); free(D);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);

}

/*bibliografia
http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
*/
