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

#include "Kernels.hpp"


int
Lucas::setupCL (void)
{
  cl_int status = 0;
  cl_device_type dType;

  if (deviceType.compare ("cpu") == 0)
    dType = CL_DEVICE_TYPE_CPU;
  else				//deviceType = "gpu" 
    {
      dType = CL_DEVICE_TYPE_GPU;
      if (isThereGPU () == false)
	{
	  std::cout << "GPU not found. Falling back to CPU device" << std::
	    endl;
	  dType = CL_DEVICE_TYPE_CPU;
	}
    }

  /*
   * Have a look at the available platforms and pick either
   * the AMD one if available or a reasonable default.
   */
  status = cl::Platform::get (&platforms);
  CHECK_OPENCL_ERROR (status, "Platform::get() failed.");

  std::vector < cl::Platform >::iterator i;
  if (platforms.size () > 0)
    {
      if (isPlatformEnabled ())
	{
	  i = platforms.begin () + platformId;
	}
      else
	{
	  for (i = platforms.begin (); i != platforms.end (); ++i)
	    {
	      if (!strcmp ((*i).getInfo < CL_PLATFORM_VENDOR > ().c_str (),
			   "Advanced Micro Devices, Inc."))
		{
		  break;
		}
	    }
	}
    }

  cl_context_properties cps[3] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) (*i) (),
    0
  };

  if (NULL == (*i) ())
    {
      sampleCommon->error ("NULL platform found so Exiting Application.");
      return SDK_FAILURE;
    }

  context = cl::Context (dType, cps, NULL, NULL, &status);
  CHECK_OPENCL_ERROR (status, "Context::Context() failed.");

  devices = context.getInfo < CL_CONTEXT_DEVICES > ();
  CHECK_OPENCL_ERROR (status, "Context::getInfo() failed.");

  std::cout << "Platform :" << (*i).getInfo < CL_PLATFORM_VENDOR >
    ().c_str () << "\n";
  int deviceCount = (int) devices.size ();
  int j = 0;
  for (std::vector < cl::Device >::iterator i = devices.begin ();
       i != devices.end (); ++i, ++j)
    {
      std::cout << "Device " << j << " : ";
      std::string deviceName = (*i).getInfo < CL_DEVICE_NAME > ();
      std::cout << deviceName.c_str () << "\n";
    }
  std::cout << "\n";

  if (deviceCount == 0)
    {
      std::cerr << "No device available\n";
      return SDK_FAILURE;
    }

  if (sampleCommon->validateDeviceId (deviceId, deviceCount))
    {
      sampleCommon->error ("sampleCommon::validateDeviceId() failed");
      return SDK_FAILURE;
    }

  std::string extensions =
    devices[deviceId].getInfo < CL_DEVICE_EXTENSIONS > ();

  std::string buildOptions = std::string ("");
  // Check if cl_khr_fp64 extension is supported 
  if (strstr (extensions.c_str (), "cl_khr_fp64"))
    {
      buildOptions.append ("-D KHR_DP_EXTENSION");
    }
  else
    {
      // Check if cl_amd_fp64 extension is supported 
      if (!strstr (extensions.c_str (), "cl_amd_fp64"))
	{
	  OPENCL_EXPECTED_ERROR
	    ("Device does not support cl_amd_fp64 extension!");
	}
    }
    cl_uint localMemType;
    // Get device specific information 
    status = devices[deviceId].getInfo<cl_uint>(
             CL_DEVICE_LOCAL_MEM_TYPE,
            &localMemType);
    CHECK_OPENCL_ERROR(status, "Device::getInfo CL_DEVICE_LOCAL_MEM_TYPE) failed.");
    
    // If scratchpad is available then update the flag 
    if(localMemType != CL_LOCAL)
	  OPENCL_EXPECTED_ERROR ("Device does not support local memory.");

    // Get Device specific Information 
    status = devices[deviceId].getInfo<size_t>(
              CL_DEVICE_MAX_WORK_GROUP_SIZE, 
              &maxWorkGroupSize);

    CHECK_OPENCL_ERROR(status, "Device::getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE) failed.");
    if(threads > maxWorkGroupSize)
	  OPENCL_EXPECTED_ERROR ("Device does not support threads.");
    
    status = devices[deviceId].getInfo<cl_uint>(
             CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
             &maxDimensions);
    CHECK_OPENCL_ERROR(status, "Device::getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS) failed.");
    

    maxWorkItemSizes = (size_t*)malloc(maxDimensions * sizeof(size_t));
    
    std::vector<size_t> workItems = devices[deviceId].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

    for(cl_uint i = 0; i < maxDimensions; ++i)
        maxWorkItemSizes[i] = workItems[i];

    status = devices[deviceId].getInfo<cl_ulong>(
             CL_DEVICE_LOCAL_MEM_SIZE,
             &totalLocalMemory);
    CHECK_OPENCL_ERROR(status, "Device::getInfo(CL_DEVICE_LOCAL_MEM_SIZES) failed.");

  // Set command queue properties
  cl_command_queue_properties prop = 0;
  if (!eAppGFLOPS)
    prop |= CL_QUEUE_PROFILING_ENABLE;

  commandQueue = cl::CommandQueue (context, devices[deviceId], prop, &status);
  CHECK_OPENCL_ERROR (status, "CommandQueue::CommandQueue() failed.");

  // Set Presistent memory only for AMD platform
  cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
  if (isAmdPlatform ())
    inMemFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;

  device.push_back (devices[deviceId]);

  // create a CL program using the kernel source
  streamsdk::SDKFile kernelFile;
  std::string kernelPath = sampleCommon->getPath ();

  kernelPath.append ("Kernels.cl");
  if (!kernelFile.open (kernelPath.c_str ()))
  {
      std::cout << "Failed to load kernel file : " << kernelPath <<
      std::endl;
      return SDK_FAILURE;
  }
  cl::Program::Sources programSource (1,
					  std::make_pair (kernelFile.
							  source ().data (),
							  kernelFile.
							  source ().size ()));

  program = cl::Program (context, programSource, &status);
  CHECK_OPENCL_ERROR (status, "Program::Program(Source) failed.");

  std::string flagsStr = std::string ("");

  status = program.build (device, flagsStr.c_str ());

  if (status != CL_SUCCESS)
    {
      if (status == CL_BUILD_PROGRAM_FAILURE)
	{
	  std::string str =
	    program.getBuildInfo < CL_PROGRAM_BUILD_LOG > (devices[deviceId]);

	  std::cout << " \n\t\t\tBUILD LOG\n";
	  std::cout << " ************************************************\n";
	  std::cout << str << std::endl;
	  std::cout << " ************************************************\n";
	}
    }
  CHECK_OPENCL_ERROR (status, "Program::build() failed.");

  // Create kernel  

  // If local memory is present then use the specific kernel 
  mul_kernel = cl::Kernel (program, "mul_Kernel", &status);

  CHECK_OPENCL_ERROR (status, "cl::Kernel failed.");
  status = mul_kernel.getWorkGroupInfo < cl_ulong > (devices[deviceId],
						 CL_KERNEL_LOCAL_MEM_SIZE,
						 &usedLocalMemory);
  CHECK_OPENCL_ERROR (status,
		      "Kernel::getWorkGroupInfo(CL_KERNEL_LOCAL_MEM_SIZE) failed"
		      ".(usedLocalMemory)");

  // Create normalize_kernel  

  // If local memory is present then use the specific kernel 
  normalize_kernel = cl::Kernel (program, "normalize_Kernel", &status);

  CHECK_OPENCL_ERROR (status, "cl::Kernel failed.");

  // Create normalize2_kernel  

  // If local memory is present then use the specific kernel 
  normalize2_kernel = cl::Kernel (program, "normalize2_Kernel", &status);

  CHECK_OPENCL_ERROR (status, "cl::Kernel failed.");

  return SDK_SUCCESS;
}

int
Lucas::mul_runCLKernels ()
{
  cl_int status;

  /* 
   * Kernel runs over complete output matrix with blocks of blockSize x blockSize 
   * running concurrently
   */
  size_t globalThreads = Nn / 4;
  size_t localThreads = 256;

  cl_int eventStatus = CL_QUEUED;

  // Set input data to matrix A and matrix B
  cl::Event ndrEvt;

  // Set appropriate arguments to the kernel 

  status = mul_kernel.setArg (0, Nn);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (Nn)");
  status = mul_kernel.setArg (1, g_x);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_x)");
  /* 
   * Enqueue a kernel run call.
   */

  // Each thread calculates 2 gaussian numbers 
  cl::NDRange gThreads (globalThreads);
  cl::NDRange lThreads (localThreads);

  status = commandQueue.enqueueNDRangeKernel (mul_kernel,
					      cl::NullRange,
					      gThreads, lThreads, 0, &ndrEvt);
  CHECK_OPENCL_ERROR (status, "CommandQueue::enqueueNDRangeKernel() failed.");
  status = commandQueue.flush ();
  CHECK_OPENCL_ERROR (status, "cl::CommandQueue.flush failed.");

  return SDK_SUCCESS;
}

int
Lucas::setup ()
{
  int status = setupCL ();
  if (status != SDK_SUCCESS)
    {
      if (status == SDK_EXPECTED_FAILURE)
	return SDK_EXPECTED_FAILURE;
      return SDK_FAILURE;
    }
  return SDK_SUCCESS;
}

int Lucas::genBinaryImage () { }
int Lucas::run () { }
int Lucas::verifyResults () { }
int Lucas::cleanup () { }

Lucas clLucas ("OpenCL Lucas test");

int
Lucas::normalize_runCLKernels ()
{
  cl_int status;

  /* 
   * Kernel runs over complete output matrix with blocks of blockSize x blockSize 
   * running concurrently
   */
  size_t globalThreads = Nn;
  size_t localThreads = threads;

  cl_int eventStatus = CL_QUEUED;

  // Set input data to matrix A and matrix B
  cl::Event ndrEvt;
  // Set appropriate arguments to the kernel 
  status = normalize_kernel.setArg (0, g_x);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_x)");
  status = normalize_kernel.setArg (1, threads);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (threads)");
  status = normalize_kernel.setArg (2, bigAB);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (bigAB)");
  status = normalize_kernel.setArg (3, bigAB);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (bigAB)");
  status = normalize_kernel.setArg (4, g_err);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_err)");
  status = normalize_kernel.setArg (5, g_err2);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_err2)");
  status = normalize_kernel.setArg (6, g_carry);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_carry)");
  status = normalize_kernel.setArg (7, g_inv);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_inv)");
  status = normalize_kernel.setArg (8, g_ttp);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_ttp)");
  status = normalize_kernel.setArg (9, g_ttmp);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_ttmp)");
  status = normalize_kernel.setArg (10, g_ttmpp);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_ttmpp)");
  status = normalize_kernel.setArg (11, maxerr);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (maxerr)");
  status = normalize_kernel.setArg (12, g_err_flag);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_err_flag)");
  //status = kernel.setArg (12, 1001*sizeof(int));
  status = normalize_kernel.setArg (13, (threads + 1) * sizeof (cl_int), NULL);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (carry)");
  /* 
   * Enqueue a kernel run call.
   */

  // Each thread calculates 2 gaussian numbers 
  cl::NDRange gThreads (globalThreads);
  cl::NDRange lThreads (localThreads);

  status = commandQueue.enqueueNDRangeKernel (normalize_kernel,
					      cl::NullRange,
					      gThreads, lThreads, 0, &ndrEvt);
  CHECK_OPENCL_ERROR (status, "CommandQueue::enqueueNDRangeKernel() failed.");
  status = commandQueue.flush ();
  CHECK_OPENCL_ERROR (status, "cl::CommandQueue.flush failed.");

  return SDK_SUCCESS;
}

int
Lucas::normalize2_runCLKernels ()
{
  cl_int status;

  /* 
   * Kernel runs over complete output matrix with blocks of blockSize x blockSize 
   * running concurrently
   */
  size_t globalThreads = Nn / threads;
  size_t localThreads = 256;

  cl_int eventStatus = CL_QUEUED;

  // Set input data to matrix A and matrix B
  cl::Event ndrEvt;

  // Set appropriate arguments to the kernel 
  status = normalize2_kernel.setArg (0, g_x);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_x)");
  status = normalize2_kernel.setArg (1, threads);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (threads)");
  status = normalize2_kernel.setArg (2, bigAB);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (bigAB)");
  status = normalize2_kernel.setArg (3, bigAB);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (bigAB)");
  status = normalize2_kernel.setArg (4, g_carry);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_carry)");
  status = normalize2_kernel.setArg (5, Nn);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (Nn)");
  status = normalize2_kernel.setArg (6, g_inv2);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_inv2)");
  status = normalize2_kernel.setArg (7, g_ttp2);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_ttp2)");
  status = normalize2_kernel.setArg (8, g_ttmp2);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_ttmp2)");
  status = normalize2_kernel.setArg (9, g_inv3);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_inv3)");
  status = normalize2_kernel.setArg (10, g_ttp3);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_ttp3)");
  status = normalize2_kernel.setArg (11, g_ttmp3);
  CHECK_OPENCL_ERROR (status, "cl::setArg failed. (g_ttmp3)");
  /* 
   * Enqueue a kernel run call.
   */

  // Each thread calculates 2 gaussian numbers 
  cl::NDRange gThreads (globalThreads);
  cl::NDRange lThreads (localThreads);

  status = commandQueue.enqueueNDRangeKernel (normalize2_kernel,
					      cl::NullRange,
					      gThreads, lThreads, 0, &ndrEvt);
  CHECK_OPENCL_ERROR (status, "CommandQueue::enqueueNDRangeKernel() failed.");
  status = commandQueue.flush ();
  CHECK_OPENCL_ERROR (status, "cl::CommandQueue.flush failed.");

  return SDK_SUCCESS;
}

int
Lucas::set_deviceId (int did)
{
  deviceId = did;
}

std::string prettyPrintclFFTStatus( const cl_int& status )
{
        switch( status )
        {
                case CLFFT_INVALID_GLOBAL_WORK_SIZE:
                        return "CLFFT_INVALID_GLOBAL_WORK_SIZE";
                case CLFFT_INVALID_MIP_LEVEL:
                        return "CLFFT_INVALID_MIP_LEVEL";
                case CLFFT_INVALID_BUFFER_SIZE:
                        return "CLFFT_INVALID_BUFFER_SIZE";
                case CLFFT_INVALID_GL_OBJECT:
                        return "CLFFT_INVALID_GL_OBJECT";
                case CLFFT_INVALID_OPERATION:
                        return "CLFFT_INVALID_OPERATION";
                case CLFFT_INVALID_EVENT:
                        return "CLFFT_INVALID_EVENT";
                case CLFFT_INVALID_EVENT_WAIT_LIST:
                        return "CLFFT_INVALID_EVENT_WAIT_LIST";
                case CLFFT_INVALID_GLOBAL_OFFSET:
                        return "CLFFT_INVALID_GLOBAL_OFFSET";
                case CLFFT_INVALID_WORK_ITEM_SIZE:
                        return "CLFFT_INVALID_WORK_ITEM_SIZE";
                case CLFFT_INVALID_WORK_GROUP_SIZE:
                        return "CLFFT_INVALID_WORK_GROUP_SIZE";
                case CLFFT_INVALID_WORK_DIMENSION:
                        return "CLFFT_INVALID_WORK_DIMENSION";
                case CLFFT_INVALID_KERNEL_ARGS:
                        return "CLFFT_INVALID_KERNEL_ARGS";
                case CLFFT_INVALID_ARG_SIZE:
                        return "CLFFT_INVALID_ARG_SIZE";
                case CLFFT_INVALID_ARG_VALUE:
                        return "CLFFT_INVALID_ARG_VALUE";
                case CLFFT_INVALID_ARG_INDEX:
                        return "CLFFT_INVALID_ARG_INDEX";
                case CLFFT_INVALID_KERNEL:
                        return "CLFFT_INVALID_KERNEL";
                case CLFFT_INVALID_KERNEL_DEFINITION:
                        return "CLFFT_INVALID_KERNEL_DEFINITION";
                case CLFFT_INVALID_KERNEL_NAME:
                        return "CLFFT_INVALID_KERNEL_NAME";
                case CLFFT_INVALID_PROGRAM_EXECUTABLE:
                        return "CLFFT_INVALID_PROGRAM_EXECUTABLE";
                case CLFFT_INVALID_PROGRAM:
                        return "CLFFT_INVALID_PROGRAM";
                case CLFFT_INVALID_BUILD_OPTIONS:
                        return "CLFFT_INVALID_BUILD_OPTIONS";
                case CLFFT_INVALID_BINARY:
                        return "CLFFT_INVALID_BINARY";
                case CLFFT_INVALID_SAMPLER:
                        return "CLFFT_INVALID_SAMPLER";
                case CLFFT_INVALID_IMAGE_SIZE:
                        return "CLFFT_INVALID_IMAGE_SIZE";
                case CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR:
                        return "CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR";
                case CLFFT_INVALID_MEM_OBJECT:
                        return "CLFFT_INVALID_MEM_OBJECT";
                case CLFFT_INVALID_HOST_PTR:
                        return "CLFFT_INVALID_HOST_PTR";
                case CLFFT_INVALID_COMMAND_QUEUE:
                        return "CLFFT_INVALID_COMMAND_QUEUE";
                case CLFFT_INVALID_QUEUE_PROPERTIES:
                        return "CLFFT_INVALID_QUEUE_PROPERTIES";
                case CLFFT_INVALID_CONTEXT:
                        return "CLFFT_INVALID_CONTEXT";
                case CLFFT_INVALID_DEVICE:
                        return "CLFFT_INVALID_DEVICE";
                case CLFFT_INVALID_PLATFORM:
                        return "CLFFT_INVALID_PLATFORM";
                case CLFFT_INVALID_DEVICE_TYPE:
                        return "CLFFT_INVALID_DEVICE_TYPE";
                case CLFFT_INVALID_VALUE:
                        return "CLFFT_INVALID_VALUE";
                case CLFFT_MAP_FAILURE:
                        return "CLFFT_MAP_FAILURE";
                case CLFFT_BUILD_PROGRAM_FAILURE:
                        return "CLFFT_BUILD_PROGRAM_FAILURE";
                case CLFFT_IMAGE_FORMAT_NOT_SUPPORTED:
                        return "CLFFT_IMAGE_FORMAT_NOT_SUPPORTED";
                case CLFFT_IMAGE_FORMAT_MISMATCH:
                        return "CLFFT_IMAGE_FORMAT_MISMATCH";
                case CLFFT_MEM_COPY_OVERLAP:
                        return "CLFFT_MEM_COPY_OVERLAP";
                case CLFFT_PROFILING_INFO_NOT_AVAILABLE:
                        return "CLFFT_PROFILING_INFO_NOT_AVAILABLE";
                case CLFFT_OUT_OF_HOST_MEMORY:
                        return "CLFFT_OUT_OF_HOST_MEMORY";
                case CLFFT_OUT_OF_RESOURCES:
                        return "CLFFT_OUT_OF_RESOURCES";
                case CLFFT_MEM_OBJECT_ALLOCATION_FAILURE:
                        return "CLFFT_MEM_OBJECT_ALLOCATION_FAILURE";
                case CLFFT_COMPILER_NOT_AVAILABLE:
                        return "CLFFT_COMPILER_NOT_AVAILABLE";
                case CLFFT_DEVICE_NOT_AVAILABLE:
                        return "CLFFT_DEVICE_NOT_AVAILABLE";
                case CLFFT_DEVICE_NOT_FOUND:
                        return "CLFFT_DEVICE_NOT_FOUND";
                case CLFFT_SUCCESS:
                        return "CLFFT_SUCCESS";
                case CLFFT_NOTIMPLEMENTED:
                        return "CLFFT_NOTIMPLEMENTED";
                case CLFFT_FILE_NOT_FOUND:
                        return "CLFFT_FILE_NOT_FOUND";
                case CLFFT_FILE_CREATE_FAILURE:
                        return "CLFFT_FILE_CREATE_FAILURE";
                case CLFFT_VERSION_MISMATCH:
                        return "CLFFT_VERSION_MISMATCH";
                case CLFFT_INVALID_PLAN:
                        return "CLFFT_INVALID_PLAN";
                default:
                        return "Error code not defined";
                break;
        }
}

//      This is used to either wrap an OpenCL function call, or to explicitly check a variable for an OpenCL error condition.
//      If an error occurs, we throw.
//      Note: std::runtime_error does not take unicode strings as input, so only strings supported
inline cl_int OpenCL_V_Throw ( cl_int res, const std::string& msg, size_t lineno )
{
        switch( res )
        {
                case    CL_SUCCESS:             /**< No error */
                        break;
                default:
                {
                        std::stringstream tmp;
                        tmp << "OPENCL_V_THROWERROR< ";
                        tmp << prettyPrintclFFTStatus( res );
                        tmp << " > (";
                        tmp << lineno;
                        tmp << "): ";
                        tmp << msg;
                        std::string errorm (tmp.str());
                        std::cout << errorm<< std::endl;
                        throw   std::runtime_error( errorm );
                }
        }

        return  res;
}
#define OPENCL_V_THROW(_status,_message) OpenCL_V_Throw (_status, _message, __LINE__)

int
Lucas::fft_setup (int length)
{
  clAmdFftSetupData fftSetupData;
  OPENCL_V_THROW(clAmdFftInitSetupData (&fftSetupData),"Failed to clAmdFftInitSetupData.");
  fftSetupData.debugFlags = CLFFT_DUMP_PROGRAMS;        // Dumps the FFT kernels
  // Setup the AMD FFT library.
  OPENCL_V_THROW(clAmdFftSetup (&fftSetupData),"Failed to clAmdFftSetup.");
  // Create FFT Plan
  const size_t logicalDimensions[1] = { length };
  // Create default plan.
  OPENCL_V_THROW(clAmdFftCreateDefaultPlan(&plan, context(), CLFFT_1D, logicalDimensions),"Failed to clAmdFftCreateDefaultPlan.");
  // Set double precision.
  OPENCL_V_THROW(clAmdFftSetPlanPrecision(plan, CLFFT_DOUBLE),"Failed to clAmdFftSetPlanPrecision.");
  // Set layout.
  OPENCL_V_THROW(clAmdFftSetLayout(plan, CLFFT_COMPLEX_INTERLEAVED,CLFFT_COMPLEX_INTERLEAVED),"Failed to clAmdFftSetLayout.");
  // Normalize forward transformation.
  OPENCL_V_THROW(clAmdFftSetPlanScale(plan, CLFFT_FORWARD, 1.0f/static_cast<cl_float>(length)),"Failed to clAmdFftSetPlanScale.");
  // Normalize backward transformation.
  OPENCL_V_THROW(clAmdFftSetPlanScale(plan, CLFFT_BACKWARD, 1.0f),"Failed to clAmdFftSetPlanScale.");
  // In-place FFT.
  OPENCL_V_THROW(clAmdFftSetResultLocation(plan, CLFFT_INPLACE),"Failed to clAmdFftSetResultLocation.");
  // Set number of transformations per plan.
  OPENCL_V_THROW(clAmdFftSetPlanBatchSize(plan, 1),"Failed to clAmdFftSetPlanBatchSize.");
  // BakePlan
  OPENCL_V_THROW(clAmdFftBakePlan(plan, 1, &commandQueue(), NULL, NULL),"Failed to clAmdFftBakePlan.");
}

int
Lucas::fft_close ()
{
  OPENCL_V_THROW(clAmdFftDestroyPlan (&plan),"Failed to clAmdFftDestroyPlan.");
  OPENCL_V_THROW(clAmdFftTeardown (),"Failed to clAmdFftTeardown.");
}
