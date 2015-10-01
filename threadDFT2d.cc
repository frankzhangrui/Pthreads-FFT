// Threaded two-dimensional Discrete FFT transform
// Rui Zhang
// ECE6122 Project 2

#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <stdio.h>
#include <valarray>
#include "Complex.h"
#include "InputImage.h"
using namespace std;

//Global Variables
int ImageWidth;
int ImageHeight;
Complex* ImageData;
Complex* ImageData1D;
Complex* ImageData2D;
InputImage* image;
int nThreads = 16;
valarray<int> pArray(16);
pthread_mutex_t pArraymutex;
pthread_mutex_t Imagemutex;
// You will likely need global variables indicating how
// many threads there are, and a Complex* that points to the
// 2d image being transformed.
unsigned ReverseBits(unsigned v);
void MyBarrier_Init(int N);
void MyBarrier();
void MyBarrier_Reset();
void Transpose(Complex* src, int N, int M, Complex* dst);
void Transform1D(Complex * h, int N);
void* Transform2DThread(void* v);
void Transform2D(const char* inputFN);
void InverseTransform1D(Complex*H, int N);
void* InverseTransform2DTHread(void* v);
void InverseTransform2D();

unsigned ReverseBits(unsigned v){ //  Provided to students
  unsigned n = ImageWidth; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value
  for (--n; n > 0; n >>= 1)
    {
      r <<= 1;        // Shift return value
      r |= (v & 0x1); // Merge in next bit
      v >>= 1;        // Shift reversal value
    }
  return r;
}

void MyBarrier_Set(int N)// you will likely need some parameters)
{
   pArray[N] = 1;
}
void MyBarrier_Init(){
  for(int i=0; i<nThreads; ++i) pArray[i]=0;
}

// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier(){
  int sum = pArray.sum();
  while(sum != nThreads){ }
}

void Transpose( Complex *matrix, int row, int col ){
    for ( int i = 0; i < row; i++ ){
       for ( int j = i + 1; j < col; j++ ){
           swap(matrix[i*col+j],matrix[j*col+i]);
       }
    }
}
// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.

// GRAD Students implement the following 2 functions.
// Undergrads can use the built-in barriers in pthreads.

// Call MyBarrier_Init once in main


void Transform1D(Complex* H, int N){
  Complex* reverseAll = new Complex[N];
  for(int i=0;i<N;++i) reverseAll[ReverseBits(i)] = H[i];
  memcpy(H,reverseAll,N*sizeof(Complex));
  int l = 0,j = 0,k = 0,step = 0, m = log2(N);
  Complex w, delta, temp;
  for(l = 1; l <= m; l++){
        step = 1 << l;
        w = Complex(1,0);
        delta = Complex(cos(2*M_PI/step),-sin(2*M_PI/step));
        for(j = 0; j < step/2; j++){
            for(k = 0; k < N/step; k++){
                temp = H[k* step + j +step/2]*w;
                H[k*step + step/2 + j] = H[k * step + j] - temp;
                H[k*step +j]      = H[k * step + j] + temp;
            }
            w = w * delta;
        }
  }
  delete[] reverseAll;
  
}

void InverseTransform1D(Complex*H, int N){
  Complex* reverseAll = new Complex[N];
  for(int i=0;i<N;++i) reverseAll[ReverseBits(i)] = H[i];
  memcpy(H,reverseAll,N);
  int l = 0,j = 0,k = 0,step = 0, m = log2(N);
  Complex w, delta, temp;
  for(l = 1; l <= m; l++){
        step = 1 << l;
        w = Complex(1,0);
        delta = Complex(cos(2*M_PI/step),sin(2*M_PI/step));
        for(j = 0; j < step/2; j++){
            for(k = 0; k < N/step; k++){
                temp = H[k* step + j +step/2]*w;
                H[k*step + step/2 + j] = H[k * step + j] - temp;
                H[k*step +j]      = H[k * step + j] + temp;
            }
            w = w * delta;
        }
  }
  delete[] reverseAll;
}

void* Transform2DTHread(void *v){
  // This is the thread startign point.  "v" is the thread number
  // Calculate 1d DFT for assigned rows
  // wait for all to complete
  // Calculate 1d DFT for assigned columns
  // Decrement active count and signal main if all complete
  long myID = (long)v;
  Complex* h = new Complex[ImageWidth];
  int numRow = ImageHeight/nThreads;
  for (int i = (myID*numRow) ; i<((myID+1)*numRow); ++i ){
    memcpy(h,ImageData+i*ImageWidth,ImageWidth*sizeof(Complex));
    Transform1D(h, ImageWidth);
    for (int j = 0 ; j < ImageWidth ; j++){
      pthread_mutex_lock(&Imagemutex);
      ImageData[i*ImageWidth+j] = h[j];
      pthread_mutex_unlock(&Imagemutex);
    }
  }
  pthread_mutex_lock(&pArraymutex);
  MyBarrier_Set(myID);
  pthread_mutex_unlock(&pArraymutex);
  cout<< pArray.sum()<<endl;
  return 0;
}

void* InverseTransform2DTHread(void *v){
  long myID = (long)v;
  Complex* h = new Complex[ImageWidth];
  int numRow = ImageHeight/nThreads;
  for (int i = (myID*numRow) ; i<((myID+1)*numRow); ++i ){
    memcpy(h,ImageData+i*ImageWidth,ImageWidth*sizeof(Complex));
    InverseTransform1D(h, ImageWidth);
    for (int j = 0 ; j < ImageWidth ; j++){
      pthread_mutex_lock(&Imagemutex);
      ImageData[i*ImageWidth+j] = h[j];
      pthread_mutex_unlock(&Imagemutex);
    }
  }
  pthread_mutex_lock(&pArraymutex);
  MyBarrier_Set(myID);
  pthread_mutex_unlock(&pArraymutex);
  return 0;
}

void Transform2D(const char* inputFN){ // Do the 2D transform here.
  image = new InputImage(inputFN); // Create the helper object for reading the image
  ImageWidth = image->GetWidth();
  ImageHeight = image->GetHeight();
  ImageData = new Complex[ImageWidth*ImageHeight];
  ImageData1D = new Complex[ImageWidth*ImageHeight];
  ImageData2D = new Complex[ImageWidth*ImageHeight];
  memcpy(ImageData,image->GetImageData(),ImageHeight*ImageWidth*sizeof(Complex));

  // Create 16 threads
  // Wait for all threads complete
  // Write the transformed data
  cout<<"beginning to do 1d transform"<<endl;
  for(int i = 0; i < nThreads; ++i){
    pthread_t pt; // pThread variable (output param from create)
    pthread_create(&pt, 0, Transform2DTHread, (void *)i);
  }
  MyBarrier();
  // reset barrier
  cout<<"all returned"<<endl;
  MyBarrier_Init();
  memcpy(ImageData1D,ImageData,ImageHeight*ImageWidth*sizeof(Complex));
  //InputImage image2("my1d.txt");

  image->SaveImageData("MyAfter1d.txt", ImageData1D, ImageWidth, ImageHeight);
  cout << "1d is finished" << endl;
  Transpose(ImageData,ImageWidth,ImageHeight);

  // Create the thread
  for(int i = 0; i < nThreads; ++i)
  {
    pthread_t pts;
    pthread_create(&pts, 0, Transform2DTHread, (void *)i);
  }
  MyBarrier();
  MyBarrier_Init();
  Transpose(ImageData,ImageWidth,ImageHeight);
  memcpy(ImageData2D,ImageData,ImageHeight*ImageWidth*sizeof(Complex));
  image->SaveImageData("Tower-DFT2D.txt", ImageData2D, ImageWidth, ImageHeight);
  cout << "2d is finished" << endl;
}


void InverseTransform2D(){
  for(int i = 0; i < nThreads; ++i){
    pthread_t pt; // pThread variable (output param from create)
    pthread_create(&pt, 0, InverseTransform2DTHread, (void *)i);
  }
  MyBarrier();
  cout << "1d inverse is finished" << endl;
  Transpose(ImageData,ImageWidth,ImageHeight);
  MyBarrier_Init();
  // Create the thread
  for(int i = 0; i < nThreads; ++i){
    pthread_t pts;
    pthread_create(&pts, 0, InverseTransform2DTHread, (void *)i);
  }
  MyBarrier();
  MyBarrier_Init();
  Transpose(ImageData,ImageWidth,ImageHeight);
  memcpy(ImageData2D,ImageData,ImageHeight*ImageWidth*sizeof(Complex));
  image->SaveImageData("MyAfterInverse.txt", ImageData2D, ImageWidth, ImageHeight);
  cout << "2d inverse is finished" << endl;
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  pthread_mutex_init(&pArraymutex,0);
  pthread_mutex_init(&Imagemutex,0);
  MyBarrier_Init();
  Transform2D(fn.c_str()); // Perform the transform.
  InverseTransform2D();
  delete[] ImageData;
  delete[] ImageData1D;
  delete[] ImageData2D;
  return 0;
}
