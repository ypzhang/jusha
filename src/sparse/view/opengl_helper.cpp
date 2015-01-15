namespace yac {
  
#if defined (__APPLE__) || defined(MACOSX)
   #include <GLUT/glut.h>
  //  #ifndef glutCloseFunc
  //  #define glutCloseFunc glutWMCloseFunc
  //  #endif
#else
#include <GL/freeglut.h>
#endif

bool initGL(int *argc, char **argv)
{
  //  glutInitDisplayMode( GLUT_ALPHA|GLUT_MULTISAMPLE|GLUT_RGBA|GLUT_DOUBLE);
  glutInitDisplayString(nullptr);
    // glutInit(argc, argv);
    // glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    // glutInitWindowSize(window_width, window_height);
    // glutCreateWindow("Cuda GL Interop (VBO)");
    // glutDisplayFunc(display);
    // glutKeyboardFunc(keyboard);
    // glutMotionFunc(motion);
    // glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    
    // // initialize necessary OpenGL extensions
    // glewInit();
    
    // if (! glewIsSupported("GL_VERSION_2_0 "))
    //     {
    //             fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    //                     fflush(stderr);
    //                         return false;
    //                             }

    // // default initialization
    //     glClearColor(0.0, 0.0, 0.0, 1.0);
    //         glDisable(GL_DEPTH_TEST);

    // // viewport
    //     glViewport(0, 0, window_width, window_height);

    // // projection
    //     glMatrixMode(GL_PROJECTION);
    //         glLoadIdentity();
    //             gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    return true;
    }
}
