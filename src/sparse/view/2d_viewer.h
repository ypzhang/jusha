#pragma once

// OpenGL Graphics includes
//#include <GL/glew.h>

namespace yac {
  class Mesh;
  
  class Viewer2d {
  public:
    bool init(int argc, char **argv, const Mesh &mesh);

    /* setup the frame call-back function */
    void set_frame_function() {
    }
    
    void display();

  private:
    const Mesh *m_mesh;
  };
}
