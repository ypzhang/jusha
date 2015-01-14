#pragma once
#include <fstream>


// copied from SU2 header
namespace yac {
  // https://github.com/su2code/SU2/wiki/Mesh-File
  enum class Su2ElemType: std::int8_t { Line = 3,
      TRIANGLE = 5,
      QUADRILATERAL = 9,
      TETRAHEDRAL = 10,
      HEXAHEDRAL = 12,
      WEDGE = 13,
      PYRAMID = 14
   };

  class Mesh;

  
  class Su2MeshParser {
  public:
    bool parse_file(const char *filename, Mesh &mesh);

  private:
    void process_line(const std::string &line, Mesh &mesh);
    bool is_comment(const std::string &line);
    bool is_config_dims(const std::string &line);
    bool is_config_elems(const std::string &line);
    bool is_config_points(const std::string &line);
    
    // setup imension
    bool has_dims(const std::string &line);
    void  get_dim(const std::string &line, Mesh &mesh);
    void  get_elems(const std::string &line);
    void  get_points(const std::string &line, Mesh &mesh);
    void  get_nmark(const std::string &line);
    
    void parse_elems(const std::string &line);
    void parse_points(const std::string &line, Mesh &mesh);
    void parse_nmark(const std::string &line);

    // members:
    int m_dims = 0;
    int m_elems = 0;
    int m_points = 0;
    int m_marks = 0;
    
    std::ifstream m_mesh_file;    
  };
} // namespace 
