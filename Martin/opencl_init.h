#ifndef _OPENCL_INIT_H_
#define _OPENCL_INIT_H_
#include "CL/cl.h"

#define OCL_MAX_NUM_PLATTFORMS 5
#define OCL_MAX_NUM_DEVICES_PER_PLATTFORM 100

typedef struct{
	cl_int err;
	cl_uint numPlatforms;
	cl_platform_id platforms[OCL_MAX_NUM_PLATTFORMS];
	cl_device_id devices[OCL_MAX_NUM_DEVICES_PER_PLATTFORM];
}OCL_DEVICES, *P_OCL_DEVICES;


void opencl_init(P_OCL_DEVICES p_ocl_devices);

#endif