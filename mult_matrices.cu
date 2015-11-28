#include <bits/stdc++.h>
#include <cuda.h>
#define H 5
#define W 5

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

//cuda function
__global__ void mult_dist(int *d_A, int *d_B,int *d_C){
  int i = blockIdx.y*blockDim.y+threadIdx.y;//todos los valores fila
  int j = blockIdx.x*blockDim.x+threadIdx.x;//todos los valores columna
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
  int *D = (int*)malloc(size);//resultado cuda

  llenar(A);
  llenar(B);

  start = clock();

  mult(A,B,C);

  end = clock();

  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  cout<<"\nTiempo invertido CPU ="<<fixed<<setprecision(6)<<cpu_time_used<<" s\n\n";

  cout<<"Primera matriz:\n";
  mostrar(A);
  cout<<"Segunda matriz:\n";
  mostrar(B);
  cout<<"Matriz resultante:\n";
  mostrar(C);


//------------------------------------------------------------
//multiplicacion utilizando cuda
  int *d_A,*d_B,*d_D;
  float blockSize = 32;//?
  dim3 dimBlock(blockSize, blockSize);//?
  //number of elements over threads per block
  dim3 dimGrid(ceil(W/float(blockSize)), ceil(H/float(blockSize)), 1);//?

  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_D, size);

  start = clock();
  cudaMemcpy(d_A, A, sizeof(int)*H*W, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(int)*H*W, cudaMemcpyHostToDevice);
  mult_dist<<<dimGrid, dimBlock>>>(d_A, d_B, d_D);

  cudaMemcpy(D,d_D,sizeof(int)*H*W,cudaMemcpyDeviceToHost);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  cout<<"Tiempo invertido GPU ="<<fixed<<setprecision(6)<<cpu_time_used<<" s\n\n";

  cout<<"Matriz resultante con cuda:\n";
  mostrar(D);
  free(A); free(B); free(C); free(D);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);

}
/*bibliografia
http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
https://github.com/sebas095/CUDA/blob/master/Entregas/codigo/MultMatrix.cpp
*/
