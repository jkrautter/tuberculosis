#ifndef _IMAGE_READ_H_
#define _IMAGE_READ_H_

#include <list>

using namespace std;

typedef struct
{
	float *image;
	unsigned int x_max;
	unsigned int y_max;
}IMAGE,*PIMAGE,&RIMAGE;
typedef struct
{
	float *matrix;
	unsigned int x_max;
	unsigned int y_max;
	unsigned int z_max;
}MATRIX, *PMATRIX, &RMATRIX;

PIMAGE read_bmp(char *file_name, RIMAGE destination);
PMATRIX read_matrix(char *file_names[], unsigned int num_files, RMATRIX destination);
PMATRIX read_matrix(list<char *> &file_names, RMATRIX destination);
void save_matrix(char *in_file, RMATRIX source, char *out_file);
#endif