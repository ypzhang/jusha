#pragma once
#include <fstream>

// copied from SU2 header



namespace yac {
  // https://github.com/su2code/SU2/wiki/Mesh-File
  enum class Su2ELemType { Line = 3,
      Triangle = 5,
      Quadrilateral = 9,
      Tetrahedral = 10,
      Hexahedral = 12,
      Wedge = 13,
      Pyramid = 14 };

  
  class Su2MeshParser {
  public:
    bool parse_file(const char *filename);

  private:
    void process_line(const std::string &line);
    bool is_comment(const std::string &line);
    bool is_config_dims(const std::string &line);
    bool is_config_elems(const std::string &line);
    
    // setup imension
    bool has_dims(const std::string &line);
    void  get_dim(const std::string &line);
    void  get_elems(const std::string &line);
    
    void  parse_elems(const std::string &line);


    // members:
    int m_dims = 0;
    int m_elems = 0;
    std::ifstream m_mesh_file;    
  };
} // namespace 
