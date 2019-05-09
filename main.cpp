#include <SFML/Config.hpp>
#include <iostream>
#include "app.h"

int main()
{
    std::cout << "SFML Mandelbrotr" << std::endl;
    std::cout << "SFML V" <<
        SFML_VERSION_MAJOR << "." <<
        SFML_VERSION_MINOR << "." <<
        SFML_VERSION_PATCH << std::endl;

    App app;
    app.run();
    return 0;
}
