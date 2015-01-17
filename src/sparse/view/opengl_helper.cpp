  
#if defined (__APPLE__) || defined(MACOSX)
#include <GLFW/glfw3.h>
#else
#include <GLFW/glfw3.h>
  //#include <GL/freeglut.h>
#endif


namespace yac {

  static void error_callback(int error, const char* description)
  {
    //    fputs(description, stderr);
  }
  static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
  {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
      glfwSetWindowShouldClose(window, GL_TRUE);
  }
  
  bool initGL(int *argc, char **argv)
  {
    //  glutInitDisplayMode( GLUT_ALPHA|GLUT_MULTISAMPLE|GLUT_RGBA|GLUT_DOUBLE);
    //  glutInitDisplayString(nullptr);
    
    GLFWwindow* window;
    
    /* Initialize the library */
    if (!glfwInit())
      return false;
    
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
      {
        glfwTerminate();
        return -1;
      }

    glfwSetErrorCallback(error_callback);
    
    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, key_callback);
    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
      {
        /* Render here */
        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float) height;
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glRotatef((float) glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
        glBegin(GL_TRIANGLES);
        glColor3f(1.f, 0.f, 0.f);
        glVertex3f(-0.6f, -0.4f, 0.f);
        glColor3f(0.f, 1.f, 0.f);
        glVertex3f(0.6f, -0.4f, 0.f);
        glColor3f(0.f, 0.f, 1.f);
        glVertex3f(0.f, 0.6f, 0.f);
        glEnd();
        
        /* Swap front and back buffers */
        glfwSwapBuffers(window);
        
        /* Poll for and process events */
        glfwPollEvents();
      }
    
    glfwTerminate();
    return true;
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

    }
}
