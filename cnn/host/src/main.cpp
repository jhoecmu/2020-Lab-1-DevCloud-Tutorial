/****************************************************************
 * Copyright (c) 2020~2020, 18-643 Course Staff, CMU
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:

 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.

 * The views and conclusions contained in the software and
 * documentation are those of the authors and should not be
 * interpreted as representing official policies, either expressed or
 * implied, of the FreeBSD Project.
 ****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "util643.h"
#include "instance643.h"
#include "kernel643.h"
#include "assert.h"
#include "float.h"

using namespace aocl_utils;

/* Default 4D array layout used by validation input and output.
 * See kernel.h for kernel specific layout of input, output 
 * and weights. */
#define ARRAY4(ptr,i4,i3,i2,i1,d4,d3,d2,d1) ((ptr)[(i4)*(d3)*(d2)*(d1)+(i3)*(d2)*(d1)+(i2)*(d1)+(i1)])

#define ACL_ALIGNMENT 64

void* acl_aligned_malloc (size_t size) {
    void *result = NULL;
    if (posix_memalign(&result, ACL_ALIGNMENT, size) != 0)
        printf("acl_aligned_malloc() failed.\n");
    return result;
}

void acl_aligned_free (void *ptr) {
    free (ptr);
}

#define AOCX_FILE "cnn.aocx"

#define NUM_KERNELS             1
#define NUM_KERNELS_TO_CREATE   NUM_KERNELS
#define NUM_QUEUES              NUM_KERNELS
#define NUM_QUEUES_TO_CREATE    NUM_KERNELS
#define NUM_QUEUES_TO_FINISH    NUM_KERNELS

// OpenCL runtime configuration
cl_kernel kernel[NUM_KERNELS_TO_CREATE];
cl_command_queue cmdQueue[NUM_QUEUES_TO_CREATE + 1]; // extra queue for reading output buffer
cl_event kernel_exec_event[NUM_QUEUES];

cl_mem input_buf                    = NULL;
cl_mem weight_buf                   = NULL;
cl_mem output_buf                   = NULL;

cl_program program                  = NULL;
cl_context context                  = NULL;

cl_platform_id platform             = NULL;
cl_device_id* devices               = NULL;

// Control whether the emulator should be used.
bool use_emulator                   = false;

const char *kernel_name[] = {
    "cnn",
};

cnndata_t* dt_input                     = NULL;
cnndata_t* dt_output                    = NULL;
cnndata_t* dt_weights                   = NULL;
cnndata_t* ref_input                    = NULL;
cnndata_t* ref_output                   = NULL;
cnndata_t* ref_weights                  = NULL;


unsigned num_devices = 0;

// Check the status returned by the OpenCL API functions
#define CHECK(status)                                               \
if (status != CL_SUCCESS)                                           \
{                                                                   \
    fprintf(stderr, "error %d in line %d.\n", status, __LINE__);    \
    exit(1);                                                        \
}                                                                   \

// Check the status returned by the OpenCL API functions, don't exit on error
#define CHECK_NO_EXIT(status)                                       \
if (status != CL_SUCCESS)                                           \
{                                                                   \
    fprintf(stderr, "error %d in line %d.\n", status, __LINE__);    \
}     

uint64_t batch_size = BATCH_SIZE;
layer_size  layer_params;
kernel_size kernel_params;
uint64_t num_elem_inputs;
uint64_t num_elem_weights;
uint64_t num_elem_outputs;

double compute_kernel_execution_time(cl_event &event, double &start_d, double &end_d)
{
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,      sizeof(cl_ulong), &end,     NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,    sizeof(cl_ulong), &start,   NULL);

    start_d = (double)1.0e-9 * start;
    end_d   = (double)1.0e-9 * end;

    return    (double)1.0e-9 * (end - start); // nanoseconds to seconds
}

// Function prototypes
void cleanup();

void ZhangIsfpga15_1_fp(cnndata_t *input, cnndata_t *output, cnndata_t *weights);
int nearlyEqual(cnndata_t a, cnndata_t b);
void verify(cnndata_t *ref, cnndata_t *checkit);

bool init_opencl(FILE *f_out);
void init_problem();
void run();
void cleanup();

void read_params(Options* options);
void print_params();

// Entry point.
int main(int argc, char **argv) {

    /*------------------------------------------------------------------------------------
     * Parse command line arguments
     *------------------------------------------------------------------------------------
     */
    Options options(argc, argv);

    // Optional argument to specify whether the emulator should be used.
    if(options.has("emulator")) {
        use_emulator = options.get<bool>("emulator");
    }

    // Take inputs
    read_params(&options);
    print_params();

    FILE *f_out = stdout;

    // Initialize OpenCL.
    if(!init_opencl(f_out)) {
        return -1;
    }

    // Initialize the problem data.
    init_problem();

    // Run the kernel.
    run();

    // Free the resources allocated
    cleanup();

    return 0;
}

void read_params(Options* options) {
    // Set default parameters
    layer_params.K_wts = K_WTS; layer_params.S_wts = S_WTS;
    layer_params.R_ofm = R_OFM; layer_params.C_ofm = C_OFM; layer_params.M_ofm = M_OFM;
    layer_params.N_ifm = N_IFM;

    kernel_params.Tm = TM;
    kernel_params.Tr = TR;
    kernel_params.Tc = TC;
    kernel_params.Tn = TN;

    // Read Kernel Params
    if (options->has("tm")) {
        if (!FIX_TM) {      
            kernel_params.Tm = options->get<uint64_t>("tm");
        } else {
            printf("tm is fixed by kernel643.h.\n");
        }
    }
    if (options->has("tr")) {
        if (!FIX_TR) {      
            kernel_params.Tr = options->get<uint64_t>("tr");
        } else {
            printf("tr is fixed by kernel643.h.\n");
        }
    }
    if (options->has("tc")) {
        if (!FIX_TC) {      
            kernel_params.Tc = options->get<uint64_t>("tc");
        } else {
            printf("tc is fixed by kernel643.h.\n");
        }
    }
    if (options->has("tn")) {
        if (!FIX_TN) {      
            kernel_params.Tn = options->get<uint64_t>("tn");
        } else {
            printf("tn is fixed by kernel643.h.\n");
        }
    }

    // Read Layer Params
    if (options->has("k")) {
        if (!FIX_K) {      
            layer_params.K_wts = options->get<uint64_t>("k");
        } else {
            printf("k is fixed by kernel643.h.\n");
        }
    }
    if (options->has("s")) {
        if (!FIX_S) {      
            layer_params.S_wts = options->get<uint64_t>("s");
        } else {
            printf("s is fixed by kernel643.h.\n");
        }
    }
    if (options->has("rofm")) {
        if (!FIX_R) {      
            layer_params.R_ofm = options->get<uint64_t>("rofm");
        } else {
            printf("rofm is fixed by kernel643.h.\n");
        }
    }
    if (options->has("cofm")) {
        if (!FIX_C) {      
            layer_params.C_ofm = options->get<uint64_t>("cofm");
        } else {
            printf("cofm is fixed by kernel643.h.\n");
        }
    }
    if (options->has("mofm")) {
        if (!FIX_M) {      
            layer_params.M_ofm = options->get<uint64_t>("mofm");
        } else {
            printf("mofm is fixed by kernel643.h.\n");
        }
    }
    if (options->has("nifm")) {
        if (!FIX_N) {      
            layer_params.N_ifm = options->get<uint64_t>("nifm");
        } else {
            printf("nifm is fixed by kernel643.h.\n");
        }
    }
    
    if (options->has("batch")) {
        batch_size = options->get<uint64_t>("batch");
    }

    // Calculate dependent paramters
    layer_params.R_ifm = layer_params.R_ofm * layer_params.S_wts + 
                            layer_params.K_wts - layer_params.S_wts;
    layer_params.C_ifm = layer_params.C_ofm * layer_params.S_wts + 
                            layer_params.K_wts - layer_params.S_wts;

    num_elem_inputs = batch_size * layer_params.N_ifm * layer_params.R_ifm * layer_params.C_ifm;
    num_elem_weights = layer_params.M_ofm * layer_params.N_ifm * layer_params.K_wts * layer_params.K_wts;
    num_elem_outputs = batch_size * layer_params.M_ofm * layer_params.R_ofm * layer_params.C_ofm;

}

void print_params() {
    printf("\n===== Host-CPU printing the CNN parameters ======\n\n");

    printf("Batch size: %lu\n\n", batch_size);

    printf("Layer Parameters: \nK_wts: \t%lu\tS_wts:\t%lu\nR_ofm:\t%lu\tC_ofm:\t%lu\tM_ofm:\t%lu\tN_ifm:\t%lu\n\n", 
        layer_params.K_wts, layer_params.S_wts, layer_params.R_ofm, layer_params.C_ofm, layer_params.M_ofm, layer_params.N_ifm);

    printf("Kernel Parameters: \nTm: \t%lu\tTn:\t%lu\tTr:\t%lu\tTc:\t%lu\n\n", 
        kernel_params.Tm, kernel_params.Tn, kernel_params.Tr, kernel_params.Tc);    
}

// Initializes the OpenCL objects.
bool init_opencl(FILE *f_out) {
    unsigned int i;

    printf("\n===== Host-CPU setting up the OpenCL platform and device ======\n\n");

    cl_int status;

    if(!setCwdToExeDir()) {
        return false;
    }

    
    //----------------------------------------------
    // Get the OpenCL platform
    //----------------------------------------------
    if (use_emulator) {
        platform = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
    } else {
        platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    }
    if(platform == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
        return -1;
    }

    //----------------------------------------------
    // Discover and initialize the devices
    //----------------------------------------------

    cl_uint numDevices = 0;

    // Device info
    char buffer[4096];
    unsigned int buf_uint;
    int device_found = 0;

    printf("Initializing IDs\n");
    status = clGetDeviceIDs(platform,
                    CL_DEVICE_TYPE_ALL,
                    0,
                    NULL,
                    &numDevices);

    if(status == CL_SUCCESS){
        clGetPlatformInfo(platform,
                        CL_PLATFORM_VENDOR,
                        4096,
                        buffer,
                        NULL);

        if(strstr(buffer, "Intel(R)") != NULL){
                device_found = 1;
        }
        printf("%s\n", buffer);

        if(device_found){
            // Allocate enough space for each device
            devices = (cl_device_id*)
            acl_aligned_malloc (numDevices * sizeof(cl_device_id));

            // Fill in devices with clGetDeviceIDs()
            status = clGetDeviceIDs(platform,
                            CL_DEVICE_TYPE_ALL,
                            numDevices,
                            devices,
                            NULL);
        }
    }

    if(!device_found) {
        printf("Failed to find a OpenCL device\n");
        exit(1);
    }

    for (i = 0; i < numDevices; i++) {
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_NAME,
                        4096,
                        buffer,
                        NULL);
        fprintf(f_out, "\nDevice Name: %s\n", buffer);

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_VENDOR,
                        4096,
                        buffer,
                        NULL);
        fprintf(f_out, "Device Vendor: %s\n", buffer);

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(buf_uint),
                        &buf_uint,
                        NULL);
        fprintf(f_out, "Device Computing Units: %u\n", buf_uint);

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_GLOBAL_MEM_SIZE,
                        sizeof(unsigned long),
                        &buffer,
                        NULL);
        fprintf(f_out, "Global Memory Size: %lu\n", *((unsigned long*)buffer));

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                        sizeof(unsigned long),
                        &buffer,
                        NULL);
        fprintf(f_out, "Global Memory Allocation Size: %lu\n\n", *((unsigned long*)buffer));
    }


    //----------------------------------------------
    // Create a context
    //----------------------------------------------

    printf("\n===== Host-CPU setting up the OpenCL command queues ======\n\n");

    // Create a context using clCreateContext() and associate it with the device

    context = clCreateContext(
                    NULL,
                    1,
                    devices,
                    NULL,
                    NULL,
                    &status); CHECK(status);

    //----------------------------------------------
    // Create command queues
    //---------------------------------------------

    // Create a command queue using clCreateCommandQueue(),
    // and associate it with the device you want to execute on
    for(i = 0; i < NUM_QUEUES_TO_CREATE; i++) {
                    fprintf(stdout,"cmdQueue i = %d, kernel name = %s\n", i, kernel_name[i]);
                    cmdQueue[i] = clCreateCommandQueue(
                            context,
                            devices[0],
                            CL_QUEUE_PROFILING_ENABLE,
                            &status); CHECK(status);
    }

    fprintf(stdout,"cmdQueue i = %d, a queue for reading the C buffer\n", i);
    cmdQueue[i] = clCreateCommandQueue(context,
                                        devices[0],
                                        CL_QUEUE_PROFILING_ENABLE,
                                        &status); CHECK(status);

    //----------------------------------------------
    // Create device buffers
    //----------------------------------------------
    printf("\n===== Host-CPU creating arrays in the FPGA device global memory (DDR4) ======\n\n");
    // Input buffer.
    input_buf = clCreateBuffer(
            context, 
            CL_MEM_READ_ONLY,
            num_elem_inputs * sizeof(cnndata_t), 
            NULL, 
            &status); CHECK(status);

    // Weight buffer.
    weight_buf = clCreateBuffer(
            context, 
            CL_MEM_READ_ONLY,
            num_elem_weights * sizeof(cnndata_t), 
            NULL, 
            &status); CHECK(status);

    // Output buffer.
    output_buf = clCreateBuffer(
            context, 
            CL_MEM_WRITE_ONLY,
            num_elem_outputs * sizeof(cnndata_t), 
            NULL, 
            &status); CHECK(status);

    //----------------------------------------------
    // Create the program from binaries
    //----------------------------------------------
    printf("\n===== Host-CPU setting up OpenCL program and kernels ======\n\n");

    size_t binary_length;
    const unsigned char *binary;

    printf("\nAOCX file: %s\n\n", AOCX_FILE);
    // create the program using binary already compiled offline using aoc (i.e. the .aocx file)
    FILE *fp = fopen(AOCX_FILE, "rb");

    if (fp == NULL) {
        printf("Failed to open the AOCX file (fopen).\n");
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long ftell_sz = ftell(fp);
    if (ftell_sz < 0) {
        printf("ftell returns a negative value.\n");
        fclose(fp);
        return -1;
    }
    else {
        binary_length = ftell_sz;
    }
    binary = (unsigned char*) malloc(sizeof(unsigned char) * binary_length);
    assert(binary && "Malloc failed");
    rewind(fp);

    size_t fread_sz = fread((void*)binary, binary_length, 1, fp);
    if (fread_sz == 0) {
        printf("Failed to read from the AOCX file (fread).\n");
        fclose(fp);
        free(const_cast<unsigned char*>(binary));
        return -1;
    }
    fclose(fp);

    // Create a program using clCreateProgramWithBinary()
    program = clCreateProgramWithBinary(
                    context,
                    1,
                    devices,
                    &binary_length,
                    (const unsigned char **)&binary,
                    &status,
                    NULL); CHECK(status);


    //----------------------------------------------
    // Create the kernel
    //----------------------------------------------

    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(status != CL_SUCCESS) {
        char log[10000] = {0};
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 10000, log, NULL);
        printf("%s\n", log);
        CHECK(status);
    }


    for(int j = 0; j < NUM_KERNELS_TO_CREATE; j++) {
        printf("Creating kernel[%d]: %s\n", j,kernel_name[j]);
        kernel[j] = clCreateKernel(program, (const char*)kernel_name[j], &status);
        CHECK(status);
    }

    return true;
}

// Initialize the data for the problem
void init_problem() {
    printf("\n===== Host-CPU preparing matrices ======\n\n");

    unsigned long row, col, to, ti, iter;

    // Allocate memory for outputs
    if ((dt_output = (cnndata_t*)acl_aligned_malloc(num_elem_outputs * sizeof(cnndata_t))) == NULL) {
            perror("Failed malloc of output matrix");
            exit(1);
    }
    if ((ref_output = (cnndata_t*)acl_aligned_malloc(num_elem_outputs * sizeof(cnndata_t))) == NULL) {
            perror("Failed malloc of reference output matrix");
            exit(1);
    }

    // Set the actual output and reference output matrices to 0.
    for(iter=0;iter<batch_size;iter++) {
        for(row = 0; row < layer_params.R_ofm; row++) {
            for(col = 0; col < layer_params.C_ofm ; col++) {
                for(to = 0; to < layer_params.M_ofm; to++) {
                    ARRAYo(dt_output, iter, to, row, col, batch_size, layer_params.M_ofm,
                           layer_params.R_ofm, layer_params.C_ofm) = 0;
                    ARRAY4(ref_output, iter, to, row, col, batch_size, layer_params.M_ofm, 
                           layer_params.R_ofm, layer_params.C_ofm) = 0;
                }
            }
        }
    }
    
    // Allocate memory for inputs
    if ((dt_input = (cnndata_t*)acl_aligned_malloc(num_elem_inputs * sizeof(cnndata_t))) == NULL) {
            perror("Failed malloc of input matrix");
            exit(1);
    }
    if ((ref_input = (cnndata_t*)acl_aligned_malloc(num_elem_inputs * sizeof(cnndata_t))) == NULL) {
            perror("Failed malloc of input matrix");
            exit(1);
    }

    // Generate the input matrix
    for(iter=0;iter<batch_size;iter++) {
        for(row = 0; row < layer_params.R_ifm; row++) {
            for(col = 0; col < layer_params.C_ifm ; col++) {
                for(ti = 0; ti < layer_params.N_ifm; ti++) {
                    cnndata_t val=(((cnndata_t)(rand()%RANGE))/RANGE);
                    ARRAY4(ref_input, iter, ti, row, col, batch_size, layer_params.N_ifm, layer_params.R_ifm, 
                           layer_params.C_ifm) = val; 
                    ARRAYi(dt_input, iter, ti, row, col, batch_size, layer_params.N_ifm, layer_params.R_ifm, 
                           layer_params.C_ifm) = val;
                }
            }
        }
    }
    
    // Allocate memory for weights
    if ((dt_weights = (cnndata_t*)acl_aligned_malloc(num_elem_weights * sizeof(cnndata_t))) == NULL) {
            perror("Failed malloc of weights matrix");
            exit(1);
    }
    if ((ref_weights = (cnndata_t*)acl_aligned_malloc(num_elem_weights * sizeof(cnndata_t))) == NULL) {
            perror("Failed malloc of weights matrix");
            exit(1);
    }

    // Generate the weight matrix
    for(to = 0; to < layer_params.M_ofm; to++) {
        for(ti = 0; ti < layer_params.N_ifm; ti++) {
            for(row = 0; row < layer_params.K_wts; row++) {
                for(col=0; col < layer_params.K_wts; col++) {
                    cnndata_t val=(((cnndata_t)(rand()%RANGE))/RANGE);
                    ARRAY4(ref_weights, to, ti, row, col, layer_params.M_ofm, layer_params.N_ifm,
                           layer_params.K_wts, layer_params.K_wts) = val; 
                    ARRAYw(dt_weights, to, ti, row, col, layer_params.M_ofm, layer_params.N_ifm,
                           layer_params.K_wts, layer_params.K_wts) = val; 
                }
            }
        }
    }
}

void run() {
    cl_int status;
    unsigned int i;

    printf("\n===== Host-CPU transferring matrices A,B to the FPGA device global memory (DDR4) via PCIe ======\n\n");

    //----------------------------------------------
    // Write host data to device buffers
    //----------------------------------------------

    // blocking writes
    status = clEnqueueWriteBuffer(
            cmdQueue[0],
            input_buf,
            CL_TRUE,
            0,
            num_elem_inputs * sizeof(cnndata_t),
            dt_input,
            0,
            NULL,
            NULL); CHECK(status);

    status = clEnqueueWriteBuffer(
            cmdQueue[0],
            weight_buf,
            CL_TRUE,
            0,
            num_elem_weights * sizeof(cnndata_t),
            dt_weights,
            0,
            NULL,
            NULL); CHECK(status);

    status = clEnqueueWriteBuffer(
            cmdQueue[0],
            output_buf,
            CL_TRUE,
            0,
            num_elem_outputs * sizeof(cnndata_t),
            dt_output,
            0,
            NULL,
            NULL); CHECK(status);

    status = clSetKernelArg(
        kernel[0],
        0,
        sizeof(cl_mem),
        (void*)&input_buf); CHECK(status);

    status = clSetKernelArg(
        kernel[0],
        1,
        sizeof(cl_mem),
        (void*)&weight_buf); CHECK(status);

    status = clSetKernelArg(
        kernel[0],
        2,
        sizeof(cl_mem),
        (void*)&output_buf); CHECK(status);

    status = clSetKernelArg(
        kernel[0],
        3,
        sizeof(uint64_t),
        (void*)&batch_size); CHECK(status);

    status = clSetKernelArg(
        kernel[0],
        4,
        sizeof(kernel_size),
        (void*)&kernel_params); CHECK(status);

    status = clSetKernelArg(
        kernel[0],
        5,
        sizeof(layer_size),
        (void*)&layer_params); CHECK(status);

    const double start_time = getCurrentTimestamp();

    //----------------------------------------------
    // Configure the work-item structure (using only tasks atm)
    //----------------------------------------------

    const size_t global_work_size[3] = { 1, 1, 1 };
    const size_t local_work_size[3] = { 1, 1, 1 };

    //----------------------------------------------
    // Enqueue the kernel for execution
    //----------------------------------------------

    printf("\n===== Host-CPU enqeuing the OpenCL kernels to the FPGA device ======\n\n");
    const double start_time1 = getCurrentTimestamp();

    for(i = 0; i < NUM_KERNELS_TO_CREATE; i++) {
        // Alternatively, can use clEnqueueTaskKernel
        // printf("clEnqueueNDRangeKernel[%d]: %s!\n", i, kernel_name[i]);
        status = clEnqueueNDRangeKernel(
                        cmdQueue[i],
                        kernel[i],
                        3,
                        NULL,
                        global_work_size,
                        local_work_size,
                        0,
                        NULL,
                        &kernel_exec_event[i]
                        );
        CHECK(status);
    }
    // printf(" *** FPGA execution started!\n");

    for(i = 0; i < NUM_KERNELS_TO_CREATE; i++) {
        status = clFlush(cmdQueue[i]);
        CHECK(status);
    }

    for(i = 0; i < NUM_QUEUES_TO_FINISH; i++) {
        status = clFinish(cmdQueue[i]); CHECK(status);
    }
    const double start_time2 = getCurrentTimestamp();

    printf(" *** FPGA execution finished!\n");
    
    double k_start_time[NUM_QUEUES_TO_FINISH];
    double k_end_time[NUM_QUEUES_TO_FINISH];
    double k_exec_time[NUM_QUEUES_TO_FINISH];

    for (i = 0; i < NUM_QUEUES_TO_FINISH; i++) {
        k_exec_time[i] = compute_kernel_execution_time(kernel_exec_event[i], k_start_time[i], k_end_time[i]);
    }

    printf("\n===== Host-CPU transferring result matrix from the FPGA device global memory (DDR4) via PCIe ======\n\n");
    
    // Read the results back from the device, blocking read
    clEnqueueReadBuffer(
                cmdQueue[0*NUM_KERNELS_TO_CREATE], // using a special queue for reading buffer C
                output_buf,
                CL_TRUE,
                0,
                num_elem_outputs * sizeof(cnndata_t),
                dt_output,
                0,
                NULL,
                NULL); CHECK(status);

    printf("\n===== Comparing FPGA results to golden reference ======\n\n");

    // Verify results.
    {
        uint64_t iter;
        for(iter=0;iter < batch_size; iter++) { 
            ZhangIsfpga15_1_fp(&ARRAY4(ref_input, iter, 0, 0, 0,
                                       batch_size, layer_params.N_ifm, layer_params.R_ifm,
                                       layer_params.C_ifm),
                               &ARRAY4(ref_output, iter, 0, 0, 0, batch_size, layer_params.M_ofm,
                                       layer_params.R_ofm, layer_params.C_ofm),
                               ref_weights);
            verify(&ARRAY4(ref_output, iter, 0, 0, 0, batch_size, layer_params.M_ofm,
                           layer_params.R_ofm, layer_params.C_ofm),
                   &ARRAYo(dt_output, iter, 0, 0, 0, batch_size, layer_params.M_ofm,
                           layer_params.R_ofm, layer_params.C_ofm));
        }    
    }
    
    printf("\n===== Reporting measured throughput ======\n\n");
    double k_earliest_start_time = k_start_time[0];
    double k_latest_end_time     = k_end_time[0];

    for (i = 1; i < NUM_QUEUES_TO_FINISH; i++) {

        if (k_start_time[i] < k_earliest_start_time)
            k_earliest_start_time   = k_start_time[i];

        if (k_end_time[i]   > k_latest_end_time)
            k_latest_end_time       = k_end_time[i];
    }

    // IMPORTANT: we care about the finish time of drain_C, once data is drained we are done
    k_latest_end_time       = k_end_time[0];


    for(i = 0; i < NUM_QUEUES_TO_FINISH; i++) {
        printf("  Kernel execution time on FPGA: %s, \n   \t\t\t\t\t\texec time = %.5f s, start=%.5f s, end=%.5f s\n", kernel_name[i], k_exec_time[i], k_start_time[i], k_end_time[i]);
    }

    double k_overall_exec_time = k_latest_end_time - k_earliest_start_time;

    printf("\n");
    printf("  FPGA CNN exec time\t\t= %.5f s\n", k_overall_exec_time);
    //printf("       FPGA CNN exec time\t\t= %.5f s\n", start_time2-start_time1);

    // multiplied by 1.0e-9 to get G-FLOPs
    printf("\n");

    double num_operations = batch_size * (double)2.0 * layer_params.M_ofm * layer_params.R_ofm * 
        layer_params.C_ofm * layer_params.N_ifm * layer_params.K_wts * layer_params.K_wts;

    printf("  # operations = %.0f\n", num_operations );
    printf("  Throughput: %.5f GFLOPS\n", (double)1.0e-9 * num_operations / k_overall_exec_time);
    //printf("       Throughput: %.5f GFLOPS\n", (double)1.0e-9 * num_operations / (start_time2-start_time1));

    printf("\n");
    printf("DONE\n");
}

void ZhangIsfpga15_1_fp(cnndata_t *input, cnndata_t *output, cnndata_t *weights) {
    printf("Computing reference output\n");
    unsigned long row, col, to, ti;

    for(row = 0; row < layer_params.R_ofm; row++) {
        for(col = 0; col < layer_params.C_ofm; col++) {
            for(to = 0; to < layer_params.M_ofm; to++) {
                for(ti = 0; ti < layer_params.N_ifm; ti++) {
                    unsigned long i, j;
                    for(i = 0; i < layer_params.K_wts; i++) {
                        for(j = 0; j < layer_params.K_wts; j++) {
                            ARRAY4(output, 0, to, row, col, 0, layer_params.M_ofm, layer_params.R_ofm, layer_params.C_ofm) += 
                                ARRAY4(weights, to, ti, i, j, layer_params.M_ofm, layer_params.N_ifm, layer_params.K_wts, layer_params.K_wts)*
                                ARRAY4(input, 0, ti, layer_params.S_wts *row + i, layer_params.S_wts * col + j, 
                                    0, layer_params.N_ifm, layer_params.R_ifm, layer_params.C_ifm);
                        }
                    }
                }
            }
        }
    }
}

void verify(cnndata_t *ref, cnndata_t *checkit) {
    printf("Verifying\n");

    unsigned long row, col, to;

    for(to = 0; to < layer_params.M_ofm; to++) {
        for(row = 0; row < layer_params.R_ofm; row++) {
            for(col = 0; col < layer_params.C_ofm ; col++) {
                if (!(nearlyEqual((cnndata_t)ARRAYo(checkit, 0, to, row, col, 0, layer_params.M_ofm,
                                                    layer_params.R_ofm, layer_params.C_ofm),
                                  (cnndata_t)ARRAY4(ref, 0, to, row, col, 0, layer_params.M_ofm,
                                                    layer_params.R_ofm, layer_params.C_ofm)))) {
                    printf("Result does not match reference: layer=%lu, row=%lu, col=%lu\n.",
                           to, row, col);
                    exit(1);
                }
            }
        }
    }

    printf("Results correct.\n\n");
}

int nearlyEqual(cnndata_t a, cnndata_t b) {
    cnndata_t absA = fabs(a);
    cnndata_t absB = fabs(b);
    cnndata_t diff = fabs(a - b);

    if (a == b) { // shortcut, handles infinities
        return 1;
    } else if (a == 0 || b == 0 || diff < FLT_MIN) {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        return diff < (EPSILON * FLT_MIN);
    } else { // use relative error
        return diff / fmin((absA + absB), FLT_MAX) < EPSILON;
    }
}

// Free the resources allocated during initialization
void cleanup() {
    //----------------------------------------------
    // Release the OpenCL resources
    //----------------------------------------------
    int i;
    // Free resources
    for(i=0; i<NUM_KERNELS_TO_CREATE; i++) {
        clReleaseKernel(kernel[i]);
    }

    for(i=0; i<NUM_QUEUES_TO_FINISH; i++) {
        clReleaseEvent(kernel_exec_event[i]);
    }

    for(i=0; i<NUM_QUEUES_TO_CREATE; i++) {
        clReleaseCommandQueue(cmdQueue[i]);
    }

    clReleaseMemObject(input_buf);
    clReleaseMemObject(weight_buf);
    clReleaseMemObject(output_buf);

    acl_aligned_free(dt_input);
    acl_aligned_free(dt_output);
    acl_aligned_free(dt_weights);

    acl_aligned_free(ref_input);
    acl_aligned_free(ref_output);
    acl_aligned_free(ref_weights);

    clReleaseProgram(program);
    clReleaseContext(context);

    acl_aligned_free(devices);
}
