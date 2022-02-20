/*----------------------------------------------
 * riemann.c - calculo de area bajo la curva
 *----------------------------------------------
 * Sumas de Riemann para calcular la integral f(x)
 *
 * Date:  2021-09-22
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define A 1
#define B 40
#define N 10e6

double f(double x);     //La funcion a integrar
double trapezoides(double a, double b, int n, double h, double *globalSum);

int main(int argc, char* argv[]) {
  double integral;
  double a=A, b=B;
  int n=N;
  double h;
  int thread_count;

  if(argc > 1) {
    a = strtol(argv[1], NULL, 10);
    b = strtol(argv[2], NULL, 10);
    thread_count = strtol(argv[3], NULL, 10);
  }

  //---- Aproximacion de la integral
  h = (b-a)/n;
  double globalSum = 0;
#pragma omp parallel num_threads(thread_count)
  {
    int n_local = n / thread_count;
    int extras = n % thread_count;
    double a_local = a + h * n_local * omp_get_thread_num();
    double b_local = n_local * h + a_local;
    if(omp_get_thread_num() == thread_count - 1) {
      b_local += extras * h;
      n_local += extras;
    }
    integral += trapezoides(a_local,b_local,n_local, h, &globalSum);
  }


  printf("Con n = %d trapezoides, nuestra aproximacion \n",n);
  printf("de la integral de %f a %f es = %.10f\n", a,b,globalSum);

  return 0;
}/*main*/

//------------------------------------------
// trapezoides
//
// Estimar la integral mediante sumas de Riemann
// Input: a,b,n,h
// Output: integral
//------------------------------------------
double trapezoides(double a, double b, int n, double h, double *globalSum) {
  double integralPart;
  int k;

  //---- Valor inicial de la integral (valores extremos)
  integralPart = (f(a) + f(b)) / 2.0;

  for(k = 1; k <= n-1; k++) {
    integralPart += f(a + k*h);
  }
  double localSum = integralPart * h;

#pragma omp critical
  {
    *globalSum += localSum;
  }

  return localSum;
}/*trapezoides*/

//------------------------------------------
// f
//
// Funcion a ser integrada
// Input: x
//------------------------------------------
double f(double x) {
  double return_val;

  return_val = x*x;

  return return_val;
}/*f*/
