#ifndef _PATIENT_H_
#define _PATIENT_H_
#include <ctime>
#include <list>
#include "features.h"

class PATIENT
{
	unsigned int id;
	list<char*>bmp_files;
	time_t date;
	TUB_FEATURES tub_features;

};

#endif