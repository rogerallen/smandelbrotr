#include <SFML/Config.hpp>
#include <iostream>
#include "window.h"

int main()
{
    std::cout << "SFML Mandelbrotr" << std::endl;
    std::cout << "SFML V" <<
        SFML_VERSION_MAJOR << "." <<
        SFML_VERSION_MINOR << "." <<
        SFML_VERSION_PATCH << std::endl;

    Window window;
    window.run();

    return 0;
}
