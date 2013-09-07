/**********************************************************************
Copyright ©2012 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef MATRIXMULDOUBLE_H_
#define MATRIXMULDOUBLE_H_

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <SDKApplication.hpp>
#include <SDKCommon.hpp>
#include <SDKCommandArgs.hpp>
#include <SDKFile.hpp>

/**
 * Lucas 
 * Class implements OpenCL Matrix Multiplication sample
 * Derived from SDKSample base class
 */
cl::Context           context;      /**< CL context */
cl::CommandQueue commandQueue;      /**< CL command queue */

class Lucas : public SDKSample
{
//cl::Context           context;      /**< CL context */
    std::vector<cl::Device> devices;    /**< CL device list */
    std::vector<cl::Device> device;     /**< CL device to be used */
    std::vector<cl::Platform> platforms;    /**< vector of platforms */
//cl::CommandQueue commandQueue;      /**< CL command queue */
    cl::Program           program;      /**< CL program  */
    cl::Kernel             mul_kernel;      /**< CL kernel */
    cl::Kernel             normalize_kernel;      /**< CL kernel */
    cl::Kernel             normalize2_kernel;      /**< CL kernel */

    size_t       maxWorkGroupSize;      /**< Device Specific Information */
    size_t    kernelWorkGroupSize;      /**< Group Size returned by kernel */
    cl_uint         maxDimensions;
    size_t *     maxWorkItemSizes;
    cl_ulong     totalLocalMemory; 
    cl_ulong      usedLocalMemory; 
    cl_ulong availableLocalMemory; 
    cl_ulong    neededLocalMemory;
    int                iterations;      /**< Number of iterations for kernel execution */
    bool eAppGFLOPS;

public:
    /** 
     * Constructor 
     * Initialize member variables
     * @param name name of sample (const char*)
     */
    Lucas(const char* name)
        : SDKSample(name)
    {
    }

    /**
     * Allocate and initialize host memory array with random values
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int setupLucas();

    /**
     * Override from SDKSample, Generate binary image of given kernel 
     * and exit application
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int genBinaryImage();

    /**
     * OpenCL related initialisations. 
     * Set up Context, Device list, Command Queue, Memory buffers
     * Build CL kernel program executable
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int setupCL();

    /**
     * Set values for kernels' arguments, enqueue calls to the kernels
     * on to the command queue, wait till end of kernel execution.
     * Get kernel start and end time if timing is enabled
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int mul_runCLKernels();
    int normalize_runCLKernels();
    int normalize2_runCLKernels();

    /**
     * Override from SDKSample, adjust width and height 
     * of execution domain, perform all sample setup
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int setup();

    /**
     * Override from SDKSample
     * Run OpenCL Matrix Multiplication
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int run();

    /**
     * Override from SDKSample
     * Cleanup memory allocations
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int cleanup();

    /**
     * Override from SDKSample
     * Verify against reference implementation
     * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int verifyResults();

    int set_deviceId (int did);
    int fft_setup (int length);
    int fft_close ();
};

#endif

cl::Buffer g_ttp, g_ttmp, g_ttmpp;
cl::Buffer g_inv, g_inv2, g_inv3;
cl::Buffer g_ttp2, g_ttmp2, g_ttp3, g_ttmp3;
cl::Buffer g_x, g_err, g_err2;
cl::Buffer g_carry;
cl::Context m_context;
std::vector < cl::Device > m_devices;
cl::Program m_program;
cl::CommandQueue m_cmdQueue;

#include <clAmdFft.h>
clAmdFftPlanHandle plan;
