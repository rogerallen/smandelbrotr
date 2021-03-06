#include "app.h"
#include <iostream>
#include <string>

void usage()
{
    std::cout << "smandelbrotr - SDL2 CUDA OpenGL Mandelbrot explorer.\n";
    std::cout << "             - by Roger Allen (rallen@gmail.com)\n";
    std::cout << "             - https://github.com/rogerallen/smandelbrotr\n";
    std::cout << "options:\n";
    std::cout << "  -d N       - select cuda device number N. (default = 0)\n";
    std::cout << "  -p PATH    - path to shaders & compute kernel. (default = '.')\n";
    std::cout << "               (typically this should be the source directory)\n";
    std::cout << "  -h         - this message.\n";
}

int main(int argc, char *argv[])
{
    std::string shaderPath = "./src";
    int cudaDevice = 0;

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            if (argv[i][1] == 'h') {
                usage();
                return 0;
            }
            else if (argv[i][1] == 'd') {
                cudaDevice = std::stoi(argv[++i]);
            }
            else if (argv[i][1] == 'p') {
                shaderPath = argv[++i];
            }
            else {
                std::cerr << "ERROR: unknown option -" << argv[i][1] << std::endl;
                usage();
                return 1;
            }
        }
    }
    App app;
    app.run(cudaDevice, shaderPath);
    return 0;
}
