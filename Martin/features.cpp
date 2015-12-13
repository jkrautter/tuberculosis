#include "features.h"
#include "iostream"
#include "image_read.h"

PMATRIX get_gradient_feature(RMATRIX CPU_input_matrix, P_OCL_DEVICES p_ocl_devices)
{
	float *mx,*my,*mz;
	int size_out = (CPU_input_matrix.x_max - 2) * (CPU_input_matrix.y_max - 2) * (CPU_input_matrix.z_max - 2);
	int size_in = CPU_input_matrix.x_max * CPU_input_matrix.y_max * CPU_input_matrix.z_max;
	mx = new float[size_out];
	my = new float[size_out];
	mz = new float[size_out];
	PMATRIX mat_gradient = new MATRIX;
	mat_gradient->x_max = (CPU_input_matrix.x_max - 2);
	mat_gradient->y_max = (CPU_input_matrix.y_max - 2);
	mat_gradient->z_max = (CPU_input_matrix.z_max - 2);
	mat_gradient->matrix = new float[size_out];

	const char *filter_code[] = {
		"__kernel void filter(__global float *src, __global float *dst, __global float *filter)\n",
		"{\n",
		"	int xf,yf,zf;\n",
		"	float sum=0.0;\n",
		"	int x = get_global_id(0);\n",
		"	int y = get_global_id(1);\n",
		"	int z = get_global_id(2);\n",
		"	int x_max = get_global_size(0);\n",
		"	int y_max = get_global_size(1);\n",
		"	int z_max = get_global_size(2);\n",
		"	int i_out = x + (y * x_max) + (z * y_max * x_max);\n",
		"	int i_in = i_out - (z * (x_max * y_max)) +  ((x_max+2)*(y_max+2)) * (z + 1) + (x_max + 2) + 1 + (y * 2);\n",
		//"	printf(\"i_out: %d  -> i_in: %d\\n\",i_out,i_in);\n",
		"	for(zf = -1; zf < 2; zf++)\n",
		"	{\n",
		"		for(yf = -1; yf < 2; yf++)\n",
		"		{\n",
		"			for(xf = -1; xf < 2; xf++)\n",
		"			{\n",
		"				int index = i_in + (zf * (x_max+2)*(y_max+2)) + xf + (yf * (x_max + 2));\n",
		"				int indexf = (xf+1) + ((yf+1) * 3) + ((zf+1) * 3 * 3);\n",
		"				sum += filter[indexf] * src[index];\n",
		//"				if(i_out==17)\n",
		//"					printf(\"(%2d,%2d,%2d) %3d %2d\\n\",xf,yf,zf,index,indexf);\n",
		"			}\n",
		"		}\n",
		"	}\n",
		//"	int level = i / ((x_max-2)*(y_max-2));\n",
		//"	int middle = offset2 + level * (x_max * y_max) + i;\n",
		//"	printf(\"%d level %d index %d\\n\",i,level,middle);\n",
		"	dst[i_out] = sum;\n",
		//"	if(i_out < 100)",
		//"	printf(\"%d -> %f\\n\",i_out,sum);\n",
		//"	printf(\"kernel: %d -> %f  -> %f\\n\",i,src[i],dst[i]);\n"
		"}\n"
	};
	const char *gradient_code[] = {
		"__kernel void gradient(__global float *mx, __global float *my, __global float *mz, __global float *dst)\n",
		"{\n",
		"	int x = get_global_id(0);\n",
		"	int y = get_global_id(1);\n",
		"	int z = get_global_id(2);\n",
		"	int x_max = get_global_size(0);\n",
		"	int y_max = get_global_size(1);\n",
		"	int i = x + (y * x_max) + (z * y_max * x_max);\n",
		"	dst[i] = sqrt(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);\n",
		//"	if(i < 100)"
		//"	{\n",
		//"		printf(\"mx %f my %f mz %f\\n\",mx[i],my[i],mz[i]);\n",
		//"	}\n",
		"}\n"
	};

	float sobelz[27] = {
		1 / sqrt(3), 1 / sqrt(2), 1 / sqrt(3),
		1 / sqrt(2), 1, 1 / sqrt(2),
		1 / sqrt(3), 1 / sqrt(2), 1 / sqrt(3),
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		-1 / sqrt(3), -1 / sqrt(2), -1 / sqrt(3),
		-1 / sqrt(2), -1, -1 / sqrt(2),
		-1 / sqrt(3), -1 / sqrt(2), -1 / sqrt(3),
	};
	float sobelx[27] = {
		1 / sqrt(3), 0, -1 / sqrt(3),
		1 / sqrt(2), 0, -1 / sqrt(2),
		1 / sqrt(3), 0, -1 / sqrt(3),
		1 / sqrt(2), 0, -1 / sqrt(2),
		1, 0, -1,
		1 / sqrt(2), 0, -1 / sqrt(2),
		1 / sqrt(3), 0, -1 / sqrt(3),
		1 / sqrt(2), 0, -1 / sqrt(2),
		1 / sqrt(3), 0, -1 / sqrt(3),
	};
	float sobely[27] = {
		1 / sqrt(3), 1 / sqrt(2), 1 / sqrt(3),
		0, 0, 0,
		-1 / sqrt(3), -1 / sqrt(2), -1 / sqrt(3),
		1 / sqrt(2), 1, 1 / sqrt(2),
		0, 0, 0,
		-1 / sqrt(2), -1, -1 / sqrt(2),
		1 / sqrt(3), 1 / sqrt(2), 1 / sqrt(3),
		0, 0, 0,
		-1 / sqrt(3), -1 / sqrt(2), -1 / sqrt(3),

	};



	//cl_int err;
	cl_context context;
	context = clCreateContext(NULL, 1, p_ocl_devices->devices, NULL, NULL, &(p_ocl_devices->err));

	cl_program program;
	program = clCreateProgramWithSource(context, sizeof(filter_code) / sizeof(*filter_code), filter_code, NULL, &(p_ocl_devices->err));
	if (clBuildProgram(program, 1, p_ocl_devices->devices, "", NULL, NULL) != CL_SUCCESS) {
		char buffer[10240];
		clGetProgramBuildInfo(program, p_ocl_devices->devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		fprintf(stderr, "CL Compilation failed:\n%s", buffer);
		abort();
	}
	clUnloadCompiler();

	cl_mem input_buffer;
	input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*size_in, NULL, &(p_ocl_devices->err));
	cl_mem output_buffer;
	output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*size_out, NULL, &(p_ocl_devices->err));
	cl_mem filter_buffer;
	filter_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*FILTER_PIXEL_SIZE, NULL, &(p_ocl_devices->err));
	cl_mem mx_buffer;
	mx_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*size_out, NULL, &(p_ocl_devices->err));
	cl_mem my_buffer;
	my_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*size_out, NULL, &(p_ocl_devices->err));
	cl_mem mz_buffer;
	mz_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*size_out, NULL, &(p_ocl_devices->err));

	cl_kernel kernel;
	kernel = clCreateKernel(program, "filter", &(p_ocl_devices->err));
	clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer);
	clSetKernelArg(kernel, 1, sizeof(mx_buffer), &mx_buffer);
	clSetKernelArg(kernel, 2, sizeof(filter_buffer), &filter_buffer);
	
	cl_command_queue queue;
	queue = clCreateCommandQueue(context, p_ocl_devices->devices[0], 0, &(p_ocl_devices->err));

	//copy content from CPU_input to GPU_input
	clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(float)*size_in, CPU_input_matrix.matrix, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, filter_buffer, CL_TRUE, 0, sizeof(float)*FILTER_PIXEL_SIZE, sobelx, 0, NULL, NULL);
	cl_event kernel_completion;
	size_t global_work_size[3] = { CPU_input_matrix.x_max - 2, CPU_input_matrix.y_max - 2, CPU_input_matrix.z_max - 2 };
	clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, NULL, 0, NULL, &kernel_completion);
	clWaitForEvents(1, &kernel_completion);
	clReleaseEvent(kernel_completion);
		//copy content from GPU_output to CPU_output
	//clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, size_out * sizeof(float), mz, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, mx_buffer, CL_TRUE, 0, size_out * sizeof(float), mat_gradient->matrix, 0, NULL, NULL);
	save_matrix("input0.bmp", *mat_gradient, "mx");


	clReleaseKernel(kernel);
	kernel = clCreateKernel(program, "filter", &(p_ocl_devices->err));
	clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer);
	clSetKernelArg(kernel, 1, sizeof(my_buffer), &my_buffer);
	clSetKernelArg(kernel, 2, sizeof(filter_buffer), &filter_buffer);


	clEnqueueWriteBuffer(queue, filter_buffer, CL_TRUE, 0, sizeof(float)*FILTER_PIXEL_SIZE, sobely, 0, NULL, NULL);
	clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, NULL, 0, NULL, &kernel_completion);
	clWaitForEvents(1, &kernel_completion);
	clReleaseEvent(kernel_completion);
	//clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, size_out * sizeof(float), my, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, my_buffer, CL_TRUE, 0, size_out * sizeof(float), mat_gradient->matrix, 0, NULL, NULL);
	save_matrix("input0.bmp", *mat_gradient, "my");

	clReleaseKernel(kernel);
	kernel = clCreateKernel(program, "filter", &(p_ocl_devices->err));
	clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer);
	clSetKernelArg(kernel, 1, sizeof(mz_buffer), &mz_buffer);
	clSetKernelArg(kernel, 2, sizeof(filter_buffer), &filter_buffer);

	clEnqueueWriteBuffer(queue, filter_buffer, CL_TRUE, 0, sizeof(float)*FILTER_PIXEL_SIZE, sobelz, 0, NULL, NULL);
	clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, NULL, 0, NULL, &kernel_completion);
	clWaitForEvents(1, &kernel_completion);
	clReleaseEvent(kernel_completion);
	//clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, size_out * sizeof(float), mz, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, mz_buffer, CL_TRUE, 0, size_out * sizeof(float), mat_gradient->matrix, 0, NULL, NULL);
	save_matrix("input0.bmp", *mat_gradient, "mz");

	//for (int i = 0; i < size_out; i++)
	//{
	//	mat_gradient.matrix[i] = mz[i];
	//}

	clReleaseProgram(program);
	program = clCreateProgramWithSource(context, sizeof(gradient_code) / sizeof(*gradient_code), gradient_code, NULL, &(p_ocl_devices->err));
	if (clBuildProgram(program, 1, p_ocl_devices->devices, "", NULL, NULL) != CL_SUCCESS) {
		char buffer[10240];
		clGetProgramBuildInfo(program, p_ocl_devices->devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		fprintf(stderr, "CL Compilation failed:\n%s", buffer);
		abort();
	}
	clUnloadCompiler();



	clReleaseKernel(kernel);
	kernel = clCreateKernel(program, "gradient", &(p_ocl_devices->err));
	clSetKernelArg(kernel, 0, sizeof(mx_buffer), &mx_buffer);
	clSetKernelArg(kernel, 1, sizeof(my_buffer), &my_buffer);
	clSetKernelArg(kernel, 2, sizeof(mz_buffer), &mz_buffer);
	clSetKernelArg(kernel, 3, sizeof(output_buffer), &output_buffer);

	queue = clCreateCommandQueue(context, p_ocl_devices->devices[0], 0, &(p_ocl_devices->err));

	//clEnqueueWriteBuffer(queue, mx_buffer, CL_TRUE, 0, sizeof(float)*size_out, mx, 0, NULL, NULL);
	//cout << mx[0];
	//clEnqueueWriteBuffer(queue, my_buffer, CL_TRUE, 0, sizeof(float)*size_out, my, 0, NULL, NULL);
	//clEnqueueWriteBuffer(queue, mz_buffer, CL_TRUE, 0, sizeof(float)*size_out, mz, 0, NULL, NULL);
	clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, NULL, 0, NULL, &kernel_completion);
	clWaitForEvents(1, &kernel_completion);
	clReleaseEvent(kernel_completion);
	//clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, size_out * sizeof(float), mat_gradient->matrix, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, size_out * sizeof(float), mat_gradient->matrix, 0, NULL, NULL);
	save_matrix("input0.bmp", *mat_gradient, "ou");

	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
	clReleaseMemObject(filter_buffer);
	clReleaseMemObject(mx_buffer);
	clReleaseMemObject(my_buffer);
	clReleaseMemObject(mz_buffer);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);

	return mat_gradient;

}
PMATRIX get_gradient_feature2(RMATRIX CPU_input_matrix, P_OCL_DEVICES p_ocl_devices)
{
	int size_out = (CPU_input_matrix.x_max - 2) * (CPU_input_matrix.y_max - 2) * (CPU_input_matrix.z_max - 2);
	int size_in = CPU_input_matrix.x_max * CPU_input_matrix.y_max * CPU_input_matrix.z_max;

	//creating return matrix
	PMATRIX mat_gradient = new MATRIX;
	mat_gradient->x_max = (CPU_input_matrix.x_max - 2);
	mat_gradient->y_max = (CPU_input_matrix.y_max - 2);
	mat_gradient->z_max = (CPU_input_matrix.z_max - 2);
	mat_gradient->matrix = new float[size_out];

	const char *gradient_code[] = {
		"__kernel void gradient(__global float *src, __global float *dst, __global float *filterx, __global float *filtery, __global float *filterz)\n",
		"{\n",
		"	int xf,yf,zf;\n",
		"	float sumx=0.0, sumy=0.0, sumz=0.0;\n",
		"	int x = get_global_id(0);\n",
		"	int y = get_global_id(1);\n",
		"	int z = get_global_id(2);\n",
		"	int x_max = get_global_size(0);\n",
		"	int y_max = get_global_size(1);\n",
		"	int z_max = get_global_size(2);\n",
		"	int i_out = x + (y * x_max) + (z * y_max * x_max);\n",
		"	int i_in = i_out - (z * (x_max * y_max)) +  ((x_max+2)*(y_max+2)) * (z + 1) + (x_max + 2) + 1 + (y * 2);\n",
		"	for(zf = -1; zf < 2; zf++)\n",
		"	{\n",
		"		for(yf = -1; yf < 2; yf++)\n",
		"		{\n",
		"			for(xf = -1; xf < 2; xf++)\n",
		"			{\n",
		"				int index = i_in + (zf * (x_max+2)*(y_max+2)) + xf + (yf * (x_max + 2));\n",
		"				int indexf = (xf+1) + ((yf+1) * 3) + ((zf+1) * 3 * 3);\n",
		"				sumx += filterx[indexf] * src[index];\n",
		"				sumy += filtery[indexf] * src[index];\n",
		"				sumz += filterz[indexf] * src[index];\n",
		"			}\n",
		"		}\n",
		"	}\n",
		"	dst[i_out] = sqrt(sumx*sumx + sumy*sumy + sumz*sumz);\n",
		"}\n"
	};

	float sobelz[27] = {
		1 / sqrt(3), 1 / sqrt(2), 1 / sqrt(3),
		1 / sqrt(2), 1, 1 / sqrt(2),
		1 / sqrt(3), 1 / sqrt(2), 1 / sqrt(3),
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		-1 / sqrt(3), -1 / sqrt(2), -1 / sqrt(3),
		-1 / sqrt(2), -1, -1 / sqrt(2),
		-1 / sqrt(3), -1 / sqrt(2), -1 / sqrt(3),
	};
	float sobelx[27] = {
		1 / sqrt(3), 0, -1 / sqrt(3),
		1 / sqrt(2), 0, -1 / sqrt(2),
		1 / sqrt(3), 0, -1 / sqrt(3),
		1 / sqrt(2), 0, -1 / sqrt(2),
		1, 0, -1,
		1 / sqrt(2), 0, -1 / sqrt(2),
		1 / sqrt(3), 0, -1 / sqrt(3),
		1 / sqrt(2), 0, -1 / sqrt(2),
		1 / sqrt(3), 0, -1 / sqrt(3),
	};
	float sobely[27] = {
		1 / sqrt(3), 1 / sqrt(2), 1 / sqrt(3),
		0, 0, 0,
		-1 / sqrt(3), -1 / sqrt(2), -1 / sqrt(3),
		1 / sqrt(2), 1, 1 / sqrt(2),
		0, 0, 0,
		-1 / sqrt(2), -1, -1 / sqrt(2),
		1 / sqrt(3), 1 / sqrt(2), 1 / sqrt(3),
		0, 0, 0,
		-1 / sqrt(3), -1 / sqrt(2), -1 / sqrt(3),
	};


	//cl_int err;
	cl_context context;
	context = clCreateContext(NULL, 1, p_ocl_devices->devices, NULL, NULL, &(p_ocl_devices->err));

	cl_program program;
	program = clCreateProgramWithSource(context, sizeof(gradient_code) / sizeof(*gradient_code), gradient_code, NULL, &(p_ocl_devices->err));
	if (clBuildProgram(program, 1, p_ocl_devices->devices, "", NULL, NULL) != CL_SUCCESS) {
		char buffer[10240];
		clGetProgramBuildInfo(program, p_ocl_devices->devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		fprintf(stderr, "CL Compilation failed:\n%s", buffer);
		abort();
	}
	clUnloadCompiler();


	cl_mem input_buffer;
	input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*size_in, NULL, &(p_ocl_devices->err));
	cl_mem output_buffer;
	output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*size_out, NULL, &(p_ocl_devices->err));
	cl_mem filterx_buffer;
	filterx_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*FILTER_PIXEL_SIZE, NULL, &(p_ocl_devices->err));
	cl_mem filtery_buffer;
	filtery_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*FILTER_PIXEL_SIZE, NULL, &(p_ocl_devices->err));
	cl_mem filterz_buffer;
	filterz_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*FILTER_PIXEL_SIZE, NULL, &(p_ocl_devices->err));

	cl_kernel kernel;
	kernel = clCreateKernel(program, "gradient", &(p_ocl_devices->err));
	clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer);
	clSetKernelArg(kernel, 1, sizeof(output_buffer), &output_buffer);
	clSetKernelArg(kernel, 2, sizeof(filterx_buffer), &filterx_buffer);
	clSetKernelArg(kernel, 3, sizeof(filtery_buffer), &filtery_buffer);
	clSetKernelArg(kernel, 4, sizeof(filterz_buffer), &filterz_buffer);

	cl_command_queue queue;
	queue = clCreateCommandQueue(context, p_ocl_devices->devices[0], 0, &(p_ocl_devices->err));


	//copy content from CPU_input to GPU_input
	clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(float)*size_in, CPU_input_matrix.matrix, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, filterx_buffer, CL_TRUE, 0, sizeof(float)*FILTER_PIXEL_SIZE, sobelx, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, filtery_buffer, CL_TRUE, 0, sizeof(float)*FILTER_PIXEL_SIZE, sobely, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, filterz_buffer, CL_TRUE, 0, sizeof(float)*FILTER_PIXEL_SIZE, sobelz, 0, NULL, NULL);
	cl_event kernel_completion;
	size_t global_work_size[3] = { CPU_input_matrix.x_max - 2, CPU_input_matrix.y_max - 2, CPU_input_matrix.z_max - 2 };
	clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, NULL, 0, NULL, &kernel_completion);
	clWaitForEvents(1, &kernel_completion);
	clReleaseEvent(kernel_completion);
	//copy content from GPU_output to CPU_output
	//clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, size_out * sizeof(float), mz, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, size_out * sizeof(float), mat_gradient->matrix, 0, NULL, NULL);

	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
	clReleaseMemObject(filterx_buffer);
	clReleaseMemObject(filtery_buffer);
	clReleaseMemObject(filterz_buffer);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);


	return mat_gradient;
}