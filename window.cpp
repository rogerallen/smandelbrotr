#include "window.h"

void Window::run()
{
    init();
    loop();
}

void Window::init()
{
    sf::ContextSettings settings;
    settings.depthBits         = 0;
    settings.stencilBits       = 0;
    settings.antialiasingLevel = 1;
    settings.majorVersion      = 3;
    settings.minorVersion      = 3;
    m_window = new sf::RenderWindow(sf::VideoMode(800, 800),
                                    "SMandelbrotr",
                                    sf::Style::Default,
                                    settings);
    m_window->setVerticalSyncEnabled(true);

    settings = m_window->getSettings();
    std::cout << "depth bits:" << settings.depthBits << std::endl;
    std::cout << "stencil bits:" << settings.stencilBits << std::endl;
    std::cout << "antialiasing level:" << settings.antialiasingLevel << std::endl;
    std::cout << "version:" << settings.majorVersion << "." << settings.minorVersion << std::endl;

}

void Window::loop()
{
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);
    sf::Font font;
    if(!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")) {
        std::cerr << "CANNOT FONT" << std::endl;
    }
    sf::Text text;
    text.setFont(font);
    text.setString("SMFL Mandelbrotr");
    text.setCharacterSize(24);
    //text.setFillColor(sf::Color::White);
    text.setColor(sf::Color::Black);

    m_window->setActive(true);

    glClearColor(1.0,1.0,0.0,0.0);

    bool running = true;
    while (running)
    {
        sf::Event event;
        while (m_window->pollEvent(event))
        {
            if (event.type == sf::Event::Closed) {
                running = false;
            }
            else if (event.type == sf::Event::Resized) {
                resize(event.size.width, event.size.height);
            }

        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        m_window->pushGLStates();
        //m_window->clear();
        m_window->draw(shape);
        m_window->draw(text);
        m_window->popGLStates();
        m_window->display();
    }

}

void Window::resize(int width, int height)
{
    glViewport(0, 0, width, height);
}
