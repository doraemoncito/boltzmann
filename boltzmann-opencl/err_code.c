//------------------------------------------------------------------------------
//
// Name:     err_code()
//
// Purpose:  Function to output descriptions of errors for an input error code
//
//
// RETURN:   echoes the input error code
//
// HISTORY:  Written by Tim Mattson, June 2010
//
//------------------------------------------------------------------------------

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <OpenCL/opencl.h>

static char unknown_error_code[64];

const char *err_code_str(cl_int err_in) {
    switch (err_in){
        case CL_INVALID_VALUE:
			return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:
			return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:
			return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
           return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
           return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
           return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:
			return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:
			return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:
			return "CL_INVALID_MEM_OBJECT";
        case CL_OUT_OF_RESOURCES:
           return "CL_OUT_OF_RESOURCES";
        case CL_INVALID_PROGRAM_EXECUTABLE:
           return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:
			return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL:
           return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:
			return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:
			return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:
			return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
           return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
           return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_GLOBAL_OFFSET:
           return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_WORK_GROUP_SIZE:
           return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
           return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_IMAGE_SIZE:
           return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_EVENT_WAIT_LIST:
           return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_MEM_COPY_OVERLAP:
           return "CL_MEM_COPY_OVERLAP";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
           return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_HOST_MEMORY:
           return "CL_OUT_OF_HOST_MEMORY";
        default:
			sprintf(unknown_error_code, "Unknown error code (%d).", err_in);
	}

	return unknown_error_code;
}

int err_code(cl_int err_in) {
	printf("\n %s\n", err_code_str(err_in));
	return err_in;
}
