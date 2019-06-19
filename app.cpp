#include "app.h"

App::App()
{
    mSwitchFullscreen = false;
    mIsFullscreen     = false;
    mZoomOutMode      = false;
    mSaveImage        = false;
    mMouseDown        = false;
    mPrevWindowWidth  = mPrevWindowHeight = -1;

    mMonitorWidth     = -1;//FIXMEsf::VideoMode::getDesktopMode().width;
    mMonitorHeight    = -1;//FIXMEsf::VideoMode::getDesktopMode().height;

    mMouseStartX = mMouseStartY = mMouseX = mMouseY = mCenterStartX = mCenterStartY = -1;

}

void App::run()
{
    if(!init()) {
        loop();
    }
    cleanup();
}

void App::cleanup() {
    std::cout << "Exiting..." << std::endl;
    mAppMandelbrot->render();
    mAppGL->render();
}

// initialize SMFL, OpenGL, CUDA & Mandelbrot classes
// return true on error
bool App::init()
{
    if(initWindow()) {
        return true;
    }
    mAppGL = new AppGL(mAppWindow, mMonitorWidth, mMonitorHeight);
    if(AppCUDA::setDevice()) {
        return true;
    }
    mAppMandelbrot = new AppMandelbrot(mAppWindow, mAppGL);
    mAppMandelbrot->init();
    return false;
}

// initialize SDL2 window
// return true on error
bool App::initWindow()
{
    // Initialize SDL Video
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Failed to initialize SDL video" << std::endl;
        return true;
    }

    SDL_DisplayMode DM;
    SDL_GetCurrentDisplayMode(0, &DM);
    mMonitorWidth  = DM.w;
    mMonitorHeight = DM.h;

    mAppWindow = new AppWindow(WINDOW_START_WIDTH, WINDOW_START_HEIGHT);

    // Create main window
    mSDLWindow = SDL_CreateWindow(
        "SMandelbrotr",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_START_WIDTH, WINDOW_START_HEIGHT,
        SDL_WINDOW_OPENGL);
    if (mSDLWindow == NULL) {
        std::cerr << "Failed to create main window" << std::endl;
        SDL_Quit();
        return true;
    }

    // Initialize rendering context
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_PROFILE_MASK,
        SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    mSDLGLContext = SDL_GL_CreateContext(mSDLWindow);
    if (mSDLGLContext == NULL) {
        std::cerr << "Failed to create GL context" << std::endl;
        SDL_DestroyWindow(mSDLWindow);
        SDL_Quit();
        return true;
    }

    SDL_GL_SetSwapInterval(1); // Use VSYNC

    // Initialize GL Extension Wrangler (GLEW)
    GLenum err;
    glewExperimental = GL_TRUE; // Please expose OpenGL 3.x+ interfaces
    err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Failed to init GLEW" << std::endl;
        SDL_GL_DeleteContext(mSDLGLContext);
        SDL_DestroyWindow(mSDLWindow);
        SDL_Quit();
        return true;
    }

    return false;
}

void App::loop()
{
    std::cout << "Running..." << std::endl;
    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE) {
                running = false;
                break;
            }
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                }
            }
        }
        update();
        mAppMandelbrot->render();
        mAppGL->render();
        SDL_GL_SwapWindow(mSDLWindow);
    }
    /*
    bool running = true;
    while (running)
    {
        sf::Event event;
        while (mRenderWindow->pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                running = false;
            }
            else if (event.type == sf::Event::Resized) {
                resize(event.size.width, event.size.height);
            }
            else if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape) {
                    running = false;
                }
            }
            else if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    mMouseStartX = mMouseX = event.mouseButton.x;
                    mMouseStartY = mMouseY = event.mouseButton.y;
                    mCenterStartX = mAppMandelbrot->centerX();
                    mCenterStartY = mAppMandelbrot->centerY();
                    mMouseDown = true;
                }
            }
            else if (event.type == sf::Event::MouseMoved) {
                mMouseX = event.mouseMove.x;
                mMouseY = event.mouseMove.y;
            }
            else if (event.type == sf::Event::MouseButtonReleased) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    mMouseDown = false;
                }
            }
            else if (event.type == sf::Event::MouseWheelScrolled) {
                const double zoomFactor = 1.1;
                if(event.mouseWheelScroll.delta > 0) {
                    mAppMandelbrot->zoomMul(zoomFactor);
                }
                else {
                    mAppMandelbrot->zoomDiv(zoomFactor);
                }
            }
        }

        update();
        mAppMandelbrot->render();
        mAppGL->render();

        mRenderWindow->display();
    }
    */
}

void App::update()
{
    mAppGL->handleResize();
    // TODO handle fullscreen
    // TODO zoomOutMode
    if(mMouseDown) {
        double dx = mMouseX - mMouseStartX;
        double dy = mMouseY - mMouseStartY;
//#ifdef DEBUG
//        std::cerr << "dx,dy = " << dx << ", " << dy << std::endl;
//#endif
        double pixelsPerMandelSpace;
        if(mAppWindow->width() > mAppWindow->height()) {
            pixelsPerMandelSpace = mAppWindow->width() * mAppMandelbrot->zoom();
        }
        else {
            pixelsPerMandelSpace = mAppWindow->height() * mAppMandelbrot->zoom();
        }
        double mandelSpacePerPixel = 2.0 / pixelsPerMandelSpace;
        double centerDx = dx * mandelSpacePerPixel;
        double centerDy = dy * mandelSpacePerPixel;
        mAppMandelbrot->centerX(mCenterStartX - centerDx);
        mAppMandelbrot->centerY(mCenterStartY - centerDy);
    }
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
