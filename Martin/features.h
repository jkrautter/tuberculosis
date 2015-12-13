#ifndef _FEATURES_H_
#define _FEATURES_H_
#include "image_read.h"
#include "opencl_init.h"

#define FILTER_PIXEL_SIZE (3*3*3)

PMATRIX get_gradient_feature(RMATRIX CPU_input_matrix, P_OCL_DEVICES p_ocl_devices);
PMATRIX get_gradient_feature2(RMATRIX CPU_input_matrix, P_OCL_DEVICES p_ocl_devices);
class TUB_FEATURES
{


};

#endif