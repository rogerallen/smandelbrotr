#ifndef SMANDELBROTR_WINDOW_H
#define SMANDELBROTR_WINDOW_H

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <iostream>

class Window {
public:
    void run();

private:
    void init();
    void loop();
    void resize(int width, int height);

    sf::RenderWindow *m_window;
};
#endif
