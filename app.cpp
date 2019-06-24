#include "app.h"

#ifdef WIN32
// https://seabird.handmade.network/blogs/p/2460-be_aware_of_high_dpi
#pragma comment(lib, "Shcore.lib")

#include <windows.h>
#include <ShellScalingAPI.h>
#include <comdef.h>
#endif

App::App()
{
    mSwitchFullscreen = false;
    mIsFullscreen     = false;
    mZoomOutMode      = false;
    mSaveImage        = false;
    mMouseDown        = false;

    mPrevWindowWidth  = mPrevWindowHeight = -1;

    mMonitorWidth     = -1;
    mMonitorHeight    = -1;

    mMouseStartX = mMouseStartY =
        mMouseX = mMouseY = mCenterStartX = mCenterStartY = -1;

}

void App::run()
{
    std::cout << "SDL2 CUDA OpenGL Mandelbrotr" << std::endl;

    SDL_version compiled;
    SDL_version linked;

    SDL_VERSION(&compiled);
    SDL_GetVersion(&linked);
    std::cout << "We compiled against SDL version    " << int(compiled.major) << "." << int(compiled.minor) << "." << int(compiled.patch) << std::endl;
    std::cout << "We are linking against SDL version " << int(linked.major) << "." << int(linked.minor) << "." << int(linked.patch) << std::endl;

    // FIXME cuda version
    // FIXME opengl version
    // FIXME GLEW version
    // FIXME GLM version

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
#ifdef WIN32
    SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
#endif

    // Initialize SDL Video
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Failed to initialize SDL video" << std::endl;
        return true;
    }

    SDL_DisplayMode DM;
    SDL_GetCurrentDisplayMode(0, &DM);
    mMonitorWidth = DM.w;// *2.25;
    mMonitorHeight = DM.h;// *2.25;

    mAppWindow = new AppWindow(WINDOW_START_WIDTH, WINDOW_START_HEIGHT);

    // Create main window
    mSDLWindow = SDL_CreateWindow(
        "SMandelbrotr",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_START_WIDTH, WINDOW_START_HEIGHT,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
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
    static bool skipNextF = false;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE) {
                running = false;
                break;
            }
            else if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE:
                    running = false;
                    break;
                case SDLK_d:
                    std::cout << "d" << std::endl;
                    mAppMandelbrot->doublePrecision(true);
                    break;
                case SDLK_s:
                    mAppMandelbrot->doublePrecision(false);
                    break;
                case SDLK_f:
                    // getting double f events when we switch to fullscreen
                    // on switch to windowed, there is only the one f
#ifndef NDEBUG
                    std::cout << "f" << skipNextF << std::endl;
#endif
                    if(!skipNextF) {
                        mSwitchFullscreen = true;
                        skipNextF = !mIsFullscreen;
                    } else {
                        skipNextF = false;
                    }
                    break;
                case SDLK_RETURN: // FIXME
                    std::cout << "zoomOutMode NYI" << std::endl;
                    break;
                case SDLK_1:
                    mAppMandelbrot->iterMult(1);
                    break;
                case SDLK_2:
                    mAppMandelbrot->iterMult(2);
                    break;
                case SDLK_3:
                    mAppMandelbrot->iterMult(3);
                    break;
                case SDLK_4:
                    mAppMandelbrot->iterMult(4);
                    break;
                case SDLK_p:
                    std::cout << "Center = " << mAppMandelbrot->centerX() << ", " << mAppMandelbrot->centerY() << std::endl;
                    std::cout << "Zoom =   " << mAppMandelbrot->zoom() << std::endl;
                    break;
                case SDLK_w: // FIXME
                    std::cout << "saveImage NYI" << std::endl;
                    break;
                }
            }
            else if (event.type == SDL_WINDOWEVENT) {
                if ((event.window.event == SDL_WINDOWEVENT_RESIZED) ||
                    (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)) {
                        resize(event.window.data1, event.window.data2);
                }
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    mMouseStartX = mMouseX = event.button.x;
                    mMouseStartY = mMouseY = event.button.y;
                    mCenterStartX = mAppMandelbrot->centerX();
                    mCenterStartY = mAppMandelbrot->centerY();
                    mMouseDown = true;
                }
            }
            else if (event.type == SDL_MOUSEMOTION) {
                mMouseX = event.motion.x;
                mMouseY = event.motion.y;
            }
            else if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    mMouseDown = false;
                }
            }
            else if (event.type == SDL_MOUSEWHEEL) {
                const double zoomFactor = 1.1;
                if(event.wheel.y > 0) {
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
        SDL_GL_SwapWindow(mSDLWindow);
    }
}

void App::update()
{
    mAppGL->handleResize();
    // handle fullscreen
    if(mSwitchFullscreen) {
#ifndef NDEBUG
        std::cout << "switch fullscreen ";
#endif
        mSwitchFullscreen = false;
        if(mIsFullscreen) { // switch to windowed
#ifndef NDEBUG
            std::cout << "to windowed" << std::endl;
#endif
            mIsFullscreen = false;
            SDL_SetWindowFullscreen(mSDLWindow, 0);
            SDL_SetWindowSize(mSDLWindow, mPrevWindowWidth, mPrevWindowHeight);
        }
        else { // switch to fullscreen
#ifndef NDEBUG
            std::cout << std::endl;
#endif
            mIsFullscreen = true;
            mPrevWindowWidth = mAppWindow->width();
            mPrevWindowHeight = mAppWindow->height();
            SDL_SetWindowSize(mSDLWindow, mMonitorWidth, mMonitorHeight);
            SDL_SetWindowFullscreen(mSDLWindow, SDL_WINDOW_FULLSCREEN_DESKTOP); // "fake" fullscreen
        }
    }
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
