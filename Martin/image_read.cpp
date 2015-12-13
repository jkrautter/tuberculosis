#include <iostream>
#include <string>
#include "image_read.h"
#include "opencv\cv.h"
#include "opencv\highgui.h"

using namespace std;

PIMAGE read_bmp(char *file_name, RIMAGE destination)
{
	IplImage* img = NULL;
	img = cvLoadImage(file_name, CV_LOAD_IMAGE_GRAYSCALE);
	destination.x_max = img->width;
	destination.y_max = img->height;
	int step = img->widthStep;
	destination.image = new float[img->width*img->height];
	uchar *data = (uchar *)img->imageData;
	//cout << "channels: " << img->nChannels << endl;
	for (int y = 0; y < img->height; y++)
	{
		for (int x = 0; x < img->width; x++)
		{
			destination.image[y*img->width + x] = data[y*step + x];
		}
	}

	//img->height = 100;
	//img->width = 100;
	//img->widthStep = 100;
	//cvSaveImage("output.bmp", img);


	cvReleaseImage(&img);
	return &destination;
}
PMATRIX read_matrix(char *file_names[], unsigned int num_files, RMATRIX destination)
{
	destination.z_max = num_files;

	IMAGE level;
	read_bmp(file_names[0], level);
	destination.x_max = level.x_max;
	destination.y_max = level.y_max;
	destination.matrix = new float[destination.z_max*destination.y_max*destination.x_max];

	cout << "read_matrix: " << destination.x_max << " x " << destination.y_max << " x " << destination.z_max << endl;
	for (int z = 0; z < num_files; z++)
	{
		IMAGE level;
		read_bmp(file_names[z], level);

		for (int y = 0; y < destination.y_max; y++)
		{
			for (int x = 0; x < destination.x_max; x++)
			{
				destination.matrix[z *destination.x_max*destination.y_max + y*destination.x_max + x] = level.image[y*destination.x_max + x];
			}
		}
		
	}


	return &destination;
}
PMATRIX read_matrix(list<char *> &file_names, RMATRIX destination)
{
	list<char *>::iterator myListIterator;

	list<char *>::iterator pos; // Iterator
	for (pos = file_names.begin(); pos != file_names.end(); pos++)
		cout << *pos << endl;


	return &destination;
}
void save_matrix(char *in_file, RMATRIX source, char *out_file)
{
	IplImage* img = NULL;
	img = cvLoadImage(in_file, CV_LOAD_IMAGE_GRAYSCALE);
	uchar *data = (uchar *)img->imageData;
	img->width = source.x_max;
	img->height = source.y_max;
	for (int z = 0; z < source.z_max; z++)
	{
		string file(out_file);
		file += to_string(z);
		file += ".bmp";
		for (int y = 0; y < source.y_max; y++)
		{
			for (int x = 0; x < source.x_max; x++)
			{
				data[x + y*img->widthStep] = source.matrix[x + y * source.x_max + z * source.x_max *source.y_max];

			}

		}
		cvSaveImage(file.c_str(),img);
	}
}