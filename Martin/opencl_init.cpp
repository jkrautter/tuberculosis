#include "opencl_init.h"
#include <iostream>

void opencl_init(P_OCL_DEVICES p_ocl_devices)
{
	p_ocl_devices->err = clGetPlatformIDs(OCL_MAX_NUM_PLATTFORMS, p_ocl_devices->platforms, &(p_ocl_devices->numPlatforms));
	if (CL_SUCCESS == p_ocl_devices->err)
		printf("\nDetected OpenCL platforms: %d\n", p_ocl_devices->numPlatforms);
	else
		printf("\nError calling clGetPlatformIDs. Error code: %d\n", p_ocl_devices->err);

	cl_uint devices_n = 0;
	//clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n);
	clGetDeviceIDs(p_ocl_devices->platforms[p_ocl_devices->numPlatforms - 1], CL_DEVICE_TYPE_GPU, OCL_MAX_NUM_DEVICES_PER_PLATTFORM, p_ocl_devices->devices, &devices_n);

}

