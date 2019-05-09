#include "app.h"

App::App()
{
    mSwitchFullscreen = false;
    mIsFullscreen = false;
    mZoomOutMode = false;
    mSaveImage = false;
    mMouseDown = false;
    mMonitorWidth = mMonitorHeight = -1;
    mPrevWindowWidth = mPrevWindowHeight = -1;
    mMouseStartX = mMouseStartY = mCenterStartX = mCenterStartY = -1;
}

void App::run()
{
    if(!init()) {
        loop();
    }
}

// initialize SMFL, OpenGL, CUDA & Mandelbrot classes
// return true on error
bool App::init()
{
    initWindow();
    mAppGL = new AppGL(mAppWindow, mMonitorWidth, mMonitorHeight);
    /*
    if(AppCUDA::setDevice()) {
        return true;
    }
    */
    return false;
}

// initialize SFML window
void App::initWindow()
{
    mAppWindow = new AppWindow(WINDOW_START_WIDTH, WINDOW_START_HEIGHT);

    sf::ContextSettings settings;
    settings.depthBits         = 0;
    settings.stencilBits       = 0;
    settings.antialiasingLevel = 1;
    settings.majorVersion      = 3;
    settings.minorVersion      = 3;
    mRenderWindow = new sf::RenderWindow(sf::VideoMode(mAppWindow->width(), mAppWindow->height()),
                                    "SMandelbrotr",
                                    sf::Style::Default,
                                    settings);
    mRenderWindow->setVerticalSyncEnabled(true);

    settings = mRenderWindow->getSettings();
    std::cout << "depth bits:" << settings.depthBits << std::endl;
    std::cout << "stencil bits:" << settings.stencilBits << std::endl;
    std::cout << "antialiasing level:" << settings.antialiasingLevel << std::endl;
    std::cout << "version:" << settings.majorVersion << "." << settings.minorVersion << std::endl;

}

void App::loop()
{
    /*
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
    */
    mRenderWindow->setActive(true);


    bool running = true;
    while (running)
    {
        sf::Event event;
        while (mRenderWindow->pollEvent(event))
        {
            if (event.type == sf::Event::Closed) {
                running = false;
            }
            else if (event.type == sf::Event::Resized) {
                resize(event.size.width, event.size.height);
            }

        }

        update();
        // mandelbrot.render();
        mAppGL->render();

        /*
        // FIXME do this only when necessary
        mRenderWindow->pushGLStates();
        //mRenderWindow->clear();
        mRenderWindow->draw(shape);
        mRenderWindow->draw(text);
        mRenderWindow->popGLStates();
        */

        mRenderWindow->display();
    }

}

void App::update()
{
    mAppGL->handleResize();
    // TODO handle fullscreen
    // TODO zoomOutMode
    // TODO mouseDown
    // TODO saveImage
}

void App::resize(unsigned width, unsigned height)
{
    if( width > 0 && height > 0 &&
        (mAppWindow->width() != width || mAppWindow->height() != height)) {
        mAppWindow->width(width);
        mAppWindow->height(height);
    }
}
