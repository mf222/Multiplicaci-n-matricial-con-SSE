/***********************************************************
* Código creado para el curso IIC2343 - Arquitectura de 
* Computadores de la Pontificia Universidad Católica de Chile
* Multiplicación de matrices usando extensiones SIMD (SSE)
* Por M. Fernanda Sepúlveda
************************************************************/

#include <stdio.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <time.h>

//Para correr se debe ejecutar:
//g++ -mavx multiplicacion.c -o multiplicacion

//Se define el tamaño de las matrices aqui
#define N 16

//Estos son los metodos que se utilizaran
int mul_matrices(float a[][N], float b[][N], float mul[][N]);
int mul_matrices_intrin(float a[][N], float b[][N], float mul[][N]);

int main() {

	float A[N][N], B[N][N], result[N][N];
	int i,j;
	float product;

	//Creacion de las dos matrices con numeros cualquiera
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			A[i][j]=i;
			B[i][j]=i;
		}
	}

	//Se rellena de 0's la matriz resultado	
	for(i=0; i<N; ++i){
		for(j=0; j<N; ++j)
		{
			result[i][j]=0;
		}
	}


	//PRUEBA DEL PRIMER ALGORITMO: SIN USO DE INTRINSIC
	clock_t begin_alg1 = clock();
	mul_matrices(A, B, result);
	clock_t end_alg1 = clock();
	printf("\n Matriz: \n");
	for(i=0; i<N; ++i){
		for(j=0; j<N; ++j)
		{
			printf("%f  ",result[i][j]);
			if(j==N-1){
				printf("\n\n");
			}
		}
	}

	//Se transpone A para facilitar la implementación
	float A_trans[N][N];
	for(i=0; i<N; ++i){
		for(j=0; j<N; ++j)
		{
			A_trans[j][i]=A[i][j];
		}
	}
	//Nuevamente se setea a 0's la matriz de resultados
	for(i=0; i<N; ++i){
		for(j=0; j<N; ++j)
		{
			result[i][j]=0;
		}
	}

	//PRUEBA DEL SEGUNDO ALGORITMO: USO DE INTRINSICS
	clock_t begin_alg2 = clock();
	mul_matrices_intrin(A_trans, B, result);
	clock_t end_alg2 = clock();
	printf("\n Matriz: \n");
	for(i=0; i<N; ++i){
		for(j=0; j<N; ++j)
		{
			printf("%f  ",result[i][j]);
			if(j==N-1){
				printf("\n\n");
			}
		}
	}

	printf("Tiempo del algoritmo 1: %f\n", (double)(end_alg1 - begin_alg1) / CLOCKS_PER_SEC);
	printf("Tiempo del algoritmo 2: %f\n", (double)(end_alg2 - begin_alg2) / CLOCKS_PER_SEC);
	
	return 0;
}


int mul_matrices(float a[][N], float b[][N], float mul[][N]){
	int i, j, k;
	for(i=0; i<N; ++i){
		for(j=0; j<N; ++j){
			for(k=0; k<N; ++k)
			{
				mul[i][j]+=a[i][k]*b[k][j];
			}
		}
	}
	return 0;
}


int mul_matrices_intrin(float a[][N], float b[][N], float mul[][N]){
	int i, j, k;
	//Declaro los vectores que se usaran
	__m128 num1, num2, num3, num4;
	//Se inicializa uno de ellos en 0, este es el que tendrá la suma acumulada
	num4 = _mm_setzero_ps();
	for(i=0; i<N; ++i){
		for(j=0; j<N; ++j)
		{
			num4 = _mm_setzero_ps();
			for(k=0; k<N; k+=4)
			{
				num1 = _mm_loadu_ps(a[i]+k); //Se carga la fila a[i] desplazada en k columnas
				num2 = _mm_loadu_ps(b[i]+k); 
				num3 = _mm_mul_ps(num1, num2); //Se multiplican las filas
				num3 = _mm_hadd_ps(num3, num3); //Se suman horizontalmente, equivalente al producto cruz
				num4 = _mm_add_ps(num4, num3); //Se añaden a la suma acumulada
			}
			num4 = _mm_hadd_ps(num4, num4); //Se suma horizontalmente con si misma para tener el elemento de la matriz
			_mm_store_ss(&mul[i][j],num4);//Se reemplaza en la posición del arreglo resultado
			}
		}
		return 0;
	}
