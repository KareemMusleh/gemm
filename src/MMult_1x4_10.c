/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot4x4( int, double *, int, double *, int, double *, int );

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for (j = 0; j < n; j += 4) {
    for (i = 0; i < m; i++){
      AddDot1x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

typedef union
{
  __m128d v;
  double d[2];
} v2df_t;

void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     And now we use vector registers and instructions */
  int p;
  // the lines bellow will HINT to the compiler to use registers to store the doubles
  register double c_00_reg, c_01_reg, c_02_reg, c_03_reg, a_0p_reg;

  c_00_reg = 0.0;
  c_01_reg = 0.0;
  c_02_reg = 0.0;
  c_03_reg = 0.0;

  double *b_p0_ptr, *b_p1_ptr, *b_p2_ptr, *b_p3_ptr;
  
  b_p0_ptr = &B( 0, 0 );
  b_p1_ptr = &B( 0, 1 );
  b_p2_ptr = &B( 0, 2 );
  b_p3_ptr = &B( 0, 3 );

  for ( p=0; p<k; p+=4 ){
    // This time we unroll this loop
    a_0p_reg = A( 0,p );
    // we can simply increment because the matrix has a column-major ordering
    // ptr++ changes rows
    c_00_reg += a_0p_reg * *b_p0_ptr;
    c_01_reg += a_0p_reg * *b_p1_ptr;
    c_02_reg += a_0p_reg * *b_p2_ptr;
    c_03_reg += a_0p_reg * *b_p3_ptr;

    a_0p_reg = A( 0,p+1 );
    c_00_reg += a_0p_reg * *(b_p0_ptr+1);
    c_01_reg += a_0p_reg * *(b_p1_ptr+1);
    c_02_reg += a_0p_reg * *(b_p2_ptr+1);
    c_03_reg += a_0p_reg * *(b_p3_ptr+1);

    a_0p_reg = A( 0,p+2 );
    c_00_reg += a_0p_reg * *(b_p0_ptr+2);
    c_01_reg += a_0p_reg * *(b_p1_ptr+2);
    c_02_reg += a_0p_reg * *(b_p2_ptr+2);
    c_03_reg += a_0p_reg * *(b_p3_ptr+2);

    a_0p_reg = A( 0,p+3 );
    c_00_reg += a_0p_reg * *(b_p0_ptr+3);
    c_01_reg += a_0p_reg * *(b_p1_ptr+3);
    c_02_reg += a_0p_reg * *(b_p2_ptr+3);
    c_03_reg += a_0p_reg * *(b_p3_ptr+3);
  
    // We update the ptr once every four iterations
    b_p0_ptr += 4;
    b_p1_ptr += 4;
    b_p2_ptr += 4;
    b_p3_ptr += 4;
  }
  C( 0,0 ) += c_00_reg;
  C( 0,1 ) += c_01_reg;
  C( 0,2 ) += c_02_reg;
  C( 0,3 ) += c_03_reg;
}