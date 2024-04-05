#include <iostream>
#include <vector>
#include <CL/cl.h>

using namespace std;

void addVectorsUsingOpenCL(vector<int>& inputVector1, vector<int>& inputVector2, vector<int>& outputVector, int vectorSize) {
    // Obtain available platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // Obtain available GPU device
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create a context and command queue for the GPU device
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create OpenCL buffers for input and output vectors
    cl_mem bufferInputVector1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * vectorSize, inputVector1.data(), NULL);
    cl_mem bufferInputVector2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * vectorSize, inputVector2.data(), NULL);
    cl_mem bufferOutputVector = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * vectorSize, NULL, NULL);

    // Define kernel source code for vector addition
    const char* kernelSourceCode =
        "_kernel void vectorAddition(_global const int* input1, __global const int* input2, __global int* output, int size) {\n"
        "   int index = get_global_id(0);\n"
        "   if (index < size) {\n"
        "       output[index] = input1[index] + input2[index];\n"
        "   }\n"
        "}\n";

    // Create an OpenCL program from the kernel source code
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCode, NULL, NULL);

    // Build the OpenCL program
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create a kernel object from the program
    cl_kernel kernel = clCreateKernel(program, "vectorAddition", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferInputVector1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferInputVector2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferOutputVector);
    clSetKernelArg(kernel, 3, sizeof(int), &vectorSize);

    // Execute the kernel on the GPU device
    size_t globalSize = vectorSize;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // Read the result back into the output vector
    clEnqueueReadBuffer(queue, bufferOutputVector, CL_TRUE, 0, sizeof(int) * vectorSize, outputVector.data(), 0, NULL, NULL);

    // Release OpenCL resources
    clReleaseMemObject(bufferInputVector1);
    clReleaseMemObject(bufferInputVector2);
    clReleaseMemObject(bufferOutputVector);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <vector_size>" << endl;
        return 1;
    }

    int vectorSize = atoi(argv[1]); // Size of input vectors

    // Create input vectors
    vector<int> inputVector1(vectorSize);
    vector<int> inputVector2(vectorSize);
    vector<int> outputVector(vectorSize);

    // Initialize input vectors
    for (int i = 0; i < vectorSize; ++i) {
        inputVector1[i] = i + 1;
        inputVector2[i] = vectorSize - i;
    }

    // Perform vector addition using OpenCL
    addVectorsUsingOpenCL(inputVector1, inputVector2, outputVector, vectorSize);

    // Print the result
    cout << "Result of vector addition:" << endl;
    for (int i = 0; i < vectorSize; ++i) {
        cout << outputVector[i] << " ";
    }
    cout << endl;

    return 0;
}
