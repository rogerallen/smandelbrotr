#include "app.h"

#ifdef WIN32
// don't interfere with std::min,max
#define NOMINMAX
// https://seabird.handmade.network/blogs/p/2460-be_aware_of_high_dpi
#pragma comment(lib, "Shcore.lib")

#include <ShellScalingAPI.h>
#include <comdef.h>
#include <windows.h>
#endif

App::App()
{
    mAppWindow = nullptr;
    mAppGL = nullptr;
    mAppMandelbrot = nullptr;
    mSDLWindow = nullptr;

    mSwitchFullscreen = false;
    mIsFullscreen = false;
    mZoomOutMode = false;
    mSaveImage = false;
    mMouseDown = false;

    mPrevWindowWidth = mPrevWindowHeight = -1;
    mPrevWindowX = mPrevWindowY = -1;

    mMonitorWidth = mMonitorHeight = -1;

    mMouseStartX = mMouseStartY =
        mMouseX = mMouseY = mCenterStartX = mCenterStartY = -1;
}

void App::run(const int cudaDevice, const std::string &shaderPath)
{
#ifndef NDEBUG
    std::cout << "SDL2 CUDA OpenGL Mandelbrotr" << std::endl;
    std::cout << "Versions-------------+-------" << std::endl;
    SDL_version compiled;
    SDL_version linked;

    SDL_VERSION(&compiled);
    SDL_GetVersion(&linked);
    std::cout << "SDL compiled version | " << int(compiled.major) << "." << int(compiled.minor) << "." << int(compiled.patch) << std::endl;
    std::cout << "SDL linked version   | " << int(linked.major) << "." << int(linked.minor) << "." << int(linked.patch) << std::endl;

    int v;
    cudaRuntimeGetVersion(&v);
    int major = v / 1000;
    int minor = (v - 1000 * major) / 10;
    std::cout << "CUDA runtime version | " << major << "." << minor << std::endl;
    cudaRuntimeGetVersion(&v);
    major = v / 1000;
    minor = (v - 1000 * major) / 10;
    std::cout << "CUDA driver version  | " << major << "." << minor << std::endl;

    std::cout << "GLM version          | " << GLM_VERSION << std::endl;
#endif

    if (!init(cudaDevice, shaderPath)) {
        loop();
    }
    cleanup();
}

void App::cleanup()
{
#ifndef NDEBUG
    std::cout << "Exiting..." << std::endl;
#endif
    SDL_DestroyWindow(mSDLWindow);
}

// initialize SMFL, OpenGL, CUDA & Mandelbrot classes
// return true on error
bool App::init(const int cudaDevice, const std::string &shaderPath)
{
    if (initWindow()) {
        return true;
    }
    mAppGL = new AppGL(mAppWindow, mMonitorWidth, mMonitorHeight, shaderPath);
    if (AppCUDA::setDevice(cudaDevice)) {
        return true;
    }
    mAppMandelbrot = new AppMandelbrot(mAppWindow, mAppGL, shaderPath);
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
    mMonitorWidth = DM.w;
    mMonitorHeight = DM.h;

    int startDim = std::min(mMonitorWidth, mMonitorHeight) / 2;
    mAppWindow = new AppWindow(startDim, startDim);

    // Create main window
    mSDLWindow = SDL_CreateWindow(
        "SMandelbrotr",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        startDim, startDim,
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

#ifndef NDEBUG
    int major, minor;
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &major);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &minor);
    std::cout << "OpenGL version       | " << major << "." << minor << std::endl;
    std::cout << "GLEW version         | " << glewGetString(GLEW_VERSION) << std::endl;
    std::cout << "---------------------+-------" << std::endl;
#endif

    return false;
}

void App::loop()
{
#ifndef NDEBUG
    std::cout << "Running..." << std::endl;
#endif
    bool running = true;
    static Uint32 lastFrameEventTime = 0;
    const Uint32 debounceTime = 100; // 100ms

    while (running) {
        Uint32 curTime = SDL_GetTicks();
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
                    mAppMandelbrot->doublePrecision(true);
                    break;
                case SDLK_s:
                    mAppMandelbrot->doublePrecision(false);
                    break;
                case SDLK_f:
                    // getting double f events when we switch to fullscreen
                    // only on desktop linux!  So, let's slow this down to
                    // "debounce" those switches
                    if (curTime > lastFrameEventTime + debounceTime) {
                        mSwitchFullscreen = true;
                        lastFrameEventTime = curTime;
                    }
                    break;
                case SDLK_RETURN:
                    mZoomOutMode = true;
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
                case SDLK_w:
                    mSaveImage = true;
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
                if (event.wheel.y > 0) {
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

// Thanks! https://gist.github.com/wduminy/5859474
SDL_Surface *flip_surface(SDL_Surface *sfc)
{
    SDL_Surface *result = SDL_CreateRGBSurface(sfc->flags, sfc->w, sfc->h,
                                               sfc->format->BytesPerPixel * 8, sfc->format->Rmask, sfc->format->Gmask,
                                               sfc->format->Bmask, sfc->format->Amask);
    const auto pitch = sfc->pitch;
    const auto pxlength = pitch * (sfc->h - 1); // FIXED BUG
    auto pixels = static_cast<unsigned char *>(sfc->pixels) + pxlength;
    auto rpixels = static_cast<unsigned char *>(result->pixels);
    for (auto line = 0; line < sfc->h; ++line) {
        memcpy(rpixels, pixels, pitch);
        pixels -= pitch;
        rpixels += pitch;
    }
    return result;
}

void App::update()
{
    mAppGL->handleResize();

    // handle fullscreen
    if (mSwitchFullscreen) {
#ifndef NDEBUG
        std::cout << "switch fullscreen ";
#endif
        mSwitchFullscreen = false;
        if (mIsFullscreen) { // switch to windowed
#ifndef NDEBUG
            std::cout << "to windowed" << std::endl;
#endif
            mIsFullscreen = false;
            SDL_SetWindowFullscreen(mSDLWindow, 0);
            SDL_RestoreWindow(mSDLWindow); // Seemingly required for Jetson
            SDL_SetWindowSize(mSDLWindow, mPrevWindowWidth, mPrevWindowHeight);
            SDL_SetWindowPosition(mSDLWindow, mPrevWindowX, mPrevWindowY);
        }
        else { // switch to fullscreen
#ifndef NDEBUG
            std::cout << std::endl;
#endif
            mIsFullscreen = true;
            mPrevWindowWidth = mAppWindow->width();
            mPrevWindowHeight = mAppWindow->height();
            SDL_GetWindowPosition(mSDLWindow, &mPrevWindowX, &mPrevWindowY);
            SDL_SetWindowSize(mSDLWindow, mMonitorWidth, mMonitorHeight);
            SDL_SetWindowFullscreen(mSDLWindow, SDL_WINDOW_FULLSCREEN_DESKTOP); // "fake" fullscreen
        }
    }

    // handle zoomOutMode
    if (mZoomOutMode) {
        mAppMandelbrot->zoomDiv(1.1);
        if (mAppMandelbrot->zoom() < 0.5) {
            mZoomOutMode = false;
        }
    }

    if (mMouseDown) {
        double dx = mMouseX - mMouseStartX;
        double dy = mMouseY - mMouseStartY;
        //#ifdef DEBUG
        //        std::cerr << "dx,dy = " << dx << ", " << dy << std::endl;
        //#endif
        double pixelsPerMandelSpace;
        if (mAppWindow->width() > mAppWindow->height()) {
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

    // saveImage
    if (mSaveImage) {
        mSaveImage = false;
        std::cout << "Saving save.bmp" << std::endl;
        SDL_Surface *surface = SDL_CreateRGBSurfaceFrom(
            (void *)mAppGL->readPixels(),
            mAppWindow->width(), mAppWindow->height(), 32, 4 * mAppWindow->width(),
            0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
        SDL_Surface *flipped_surface = flip_surface(surface);
        SDL_SaveBMP(flipped_surface, "save.bmp");
        SDL_FreeSurface(flipped_surface);
        SDL_FreeSurface(surface);
    }
}

void App::resize(unsigned width, unsigned height)
{
    if (width > 0 && height > 0 &&
        (mAppWindow->width() != width || mAppWindow->height() != height)) {
        mAppWindow->width(width);
        mAppWindow->height(height);
    }
}
