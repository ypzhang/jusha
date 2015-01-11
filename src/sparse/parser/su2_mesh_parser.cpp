#include <iostream>
#include <sstream>
#include <cassert>
#include "./su2_mesh_parser.h"

using namespace std;
namespace yac {
  static char dims_prefix[]="NDIME=";
  static char elems_prefix[]="NELEM=";
  
  bool Su2MeshParser::parse_file(const char *filename)
  {
    m_mesh_file.open(filename);
    if (!m_mesh_file.is_open()) {
      std::cerr << "File " << filename << " can't be opened for reading" << std::endl;
      return false;
    }

    string line; 
    while (! m_mesh_file.eof() ) {
      getline(m_mesh_file, line);
      process_line(line);
    }

    m_mesh_file.close();
    return true;
  }

  void Su2MeshParser::process_line(const string &line)
  {
    if (is_comment(line)) return;
    
    if (is_config_dims(line))  get_dim(line);
    if (is_config_elems(line)) parse_elems(line);
    
  }

  bool Su2MeshParser::is_comment(const string &line)
  {
    if (line.size() > 0)
      return (line.at(0) == '%');
    return true;
  }

  bool Su2MeshParser::is_config_dims(const string &line)
  {
    if (line.size() > 0)
      return (line.compare(0, 6, dims_prefix) == 0);
    return false;
  }

  bool Su2MeshParser::is_config_elems(const string &line)
  {
    if (line.size() > 0)
      return (line.compare(0, 6, elems_prefix) == 0);
    return false;
  }


  void Su2MeshParser::get_dim(const std::string &line)
  {
    if (line.size() < sizeof(elems_prefix)) return;
    std::string number = line.substr (sizeof("NDIME="));
    std::istringstream ss(number);
    ss >> m_dims;
  }

  void Su2MeshParser::get_elems(const std::string &line)
  {
    if (line.size() < sizeof(elems_prefix)) return;
    std::string number = line.substr (sizeof(elems_prefix));
    std::istringstream ss(number);
    ss >> m_dims;
  }

  void Su2MeshParser::parse_elems(const std::string &line)
  {
    assert(!m_mesh_file.eof());
    
  }

  

} // namespace 
