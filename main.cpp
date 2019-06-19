#include <GL/glew.h>
#include <SDL.h>
#include <iostream>
#include "app.h"

int main()
{
    std::cout << "SDL2 CUDA OpenGL Mandelbrotr" << std::endl;

    SDL_version compiled;
    SDL_version linked;

    SDL_VERSION(&compiled);
    SDL_GetVersion(&linked);
    std::cout << "We compiled against SDL version    " << compiled.major << "." << compiled.minor << "." << compiled.patch << std::endl;
    std::cout << "We are linking against SDL version " << linked.major << "." << linked.minor << "." << linked.patch << std::endl;

    App app;
    app.run();
    return 0;
}
