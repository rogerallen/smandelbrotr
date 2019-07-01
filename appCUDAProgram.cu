#include "appCUDAProgram.h"
#include "cudaErrorCheck.h"

#include <fstream>
#include <iostream>

void readFile(const char *fileName, char **fileData)
{
    std::ifstream inputFile(fileName, std::ios::in | std::ios::binary | std::ios::ate);

    if (!inputFile.is_open()) {
        std::cerr << "\nerror: unable to open " << fileName << " for reading!\n";
        exit(1);
    }

    std::streampos pos = inputFile.tellg();
    size_t inputSize = (size_t)pos;
    *fileData = new char[inputSize + 1];

    inputFile.seekg(0, std::ios::beg);
    inputFile.read(*fileData, inputSize);
    inputFile.close();
    (*fileData)[inputSize] = '\x0';
}

void compileFile(nvrtcProgram **prog, const char *fileName, char *fileData)
{
    // compile
    int numCompileOptions = 0;
    char *compileParams[1];
    nvrtcErrChk(nvrtcCreateProgram(*prog, fileData, fileName, 0, NULL, NULL));
    nvrtcResult res = nvrtcCompileProgram(**prog, numCompileOptions, compileParams);
    // dump log if necessary
    size_t logSize;
    nvrtcErrChk(nvrtcGetProgramLogSize(**prog, &logSize));
    char *log = reinterpret_cast<char *>(malloc(sizeof(char) * logSize + 1));
    nvrtcErrChk(nvrtcGetProgramLog(**prog, log));
    log[logSize] = '\x0';
    if (strlen(log) >= 2) {
        std::cerr << "\n compilation log ---\n";
        std::cerr << log;
        std::cerr << "\n end log ---\n";
    }
    free(log);
    nvrtcErrChk(res);
}

AppCUDAProgram::AppCUDAProgram()
{
    mModule = new CUmodule();
}

void AppCUDAProgram::init(const std::string &fileName) 
{
    char *ptx;
    size_t ptxSize;
    char *fileData;
    readFile(fileName.c_str(), &fileData);

    nvrtcProgram *prog = new nvrtcProgram();
    compileFile(&prog, fileName.c_str(), fileData);

    nvrtcErrChk(nvrtcGetPTXSize(*prog, &ptxSize));
    ptx = reinterpret_cast<char *>(malloc(sizeof(char) * ptxSize));
    nvrtcErrChk(nvrtcGetPTX(*prog, ptx));

    nvrtcErrChk(nvrtcDestroyProgram(prog));

    cuErrChk(cuModuleLoadDataEx(mModule, ptx, 0, 0, 0));
#ifndef NDEBUG
    std::cout << "module " << fileName << " loaded.  PTX compiled size = " << ptxSize << std::endl;
#endif
    free(ptx);

}

// given the kernelName, return the CUfunction within mModule
CUfunction AppCUDAProgram::function(const std::string &kernelName) {
    CUfunction kernelAddr;
    cuErrChk(cuModuleGetFunction(&kernelAddr, *mModule, kernelName.c_str()));
    return kernelAddr;
}
