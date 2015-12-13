#include <iostream>
#include "image_read.h"
#include "opencl_init.h"
#include "features.h"
#include "opencv\cv.h"
#include <list>
using namespace std;




int main(int argc, char* argv[])
{
	IMAGE picture;
	MATRIX input_mat;
	PMATRIX gradient;
	char *files[] = { "input0.bmp", "input1.bmp","input2.bmp","input3.bmp" };
	list<char *> file_list;
	file_list.insert(file_list.end(), files[0]);
	file_list.insert(file_list.end(), files[1]);
	file_list.insert(file_list.end(), files[2]);
	file_list.insert(file_list.end(), files[3]);
	read_matrix(file_list, input_mat);

	//read_bmp("input.bmp", picture);

	read_matrix(files, 4, input_mat);

	OCL_DEVICES my_devices;
	opencl_init(&my_devices);
	long long t1 = cv::getTickCount();
	gradient = get_gradient_feature2(input_mat, &my_devices);
	long long t2 = cv::getTickCount();
	//cout << "freq: " << cvGetTickFrequency() << endl;
	cout << "time: " << (t2 - t1) / cv::getTickFrequency() << " s" << endl;

	//save_matrix("input.bmp", input_mat, "test");

	save_matrix("input.bmp", *gradient, "output");

	return 0;
}