#include "./2d_viewer.h"
#include "sparse/mesh.h"
#include "./opengl_helper.h"

namespace yac {
  bool Viewer2d::init(int argc, char **argv, const Mesh &mesh)
  {
    m_mesh = &mesh;

    // init openGL
    if (false == initGL(&argc, argv))
      {
        std::cerr << "openGL initialization error." << std::endl;
        return false;
      }
    return true;
  }
  void Viewer2d::display()
  {
  }
}
