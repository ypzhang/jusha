#include <iostream>
#include <sstream>
#include <cassert>
#include "./su2_mesh_parser.h"
#include "../mesh.h"

using namespace std;
namespace yac {
  static char dims_prefix[]="NDIME=";
  static char elems_prefix[]="NELEM=";
  static char points_prefix[]="NPOIN=";

  bool Su2MeshParser::parse_file(const char *filename, Mesh &mesh)
  {
    m_mesh_file.open(filename);
    if (!m_mesh_file.is_open()) {
      std::cerr << "File " << filename << " can't be opened for reading" << std::endl;
      return false;
    }

    string line; 
    while (! m_mesh_file.eof() ) {
      getline(m_mesh_file, line);
      //      printf("getting new line %s.\n", line.c_str());
      process_line(line, mesh);
    }

    m_mesh_file.close();
    return true;
  }

  void Su2MeshParser::process_line(const string &line, Mesh &mesh)
  {
    if (is_comment(line)) return;
    
    if (is_config_dims(line))  get_dim(line, mesh);
    else if (is_config_elems(line)) parse_elems(line);
    else if (is_config_points(line)) parse_points(line, mesh);
  }

  bool Su2MeshParser::is_comment(const string &line)
  {
    if (line.size() > 0)
      return (line.at(0) == '%');
    return true;
  }

  bool Su2MeshParser::is_config_dims(const string &line)
  {
    // printf("sizeof(dims_prefix) is %ld.\n", sizeof(dims_prefix));
    // printf("checking config dims %s %s true ? %d.\n", line.c_str(), dims_prefix, strncmp(line.c_str(), dims_prefix, sizeof(dims_prefix)));
    //    printf("checking config dims %s true ? %d.\n", line.c_str(), line.compare(0, sizeof(dims_prefix)-1, dims_prefix) == 0);
    if (line.size() > sizeof(dims_prefix))
      return (line.compare(0, sizeof(dims_prefix)-1, dims_prefix) == 0);
    return false;
  }

  bool Su2MeshParser::is_config_elems(const string &line)
  {
    //    printf("checking config elemens %s true ? %d.\n", line.c_str(), line.compare(0, sizeof(elems_prefix)-1, elems_prefix) == 0);
    if (line.size() > 0)
      return (line.compare(0, sizeof(elems_prefix)-1, elems_prefix) == 0);
    return false;
  }

  bool Su2MeshParser::is_config_points(const std::string &line)
  {
    //    printf("checking config points %s true ? %d.\n", line.c_str(), line.compare(0, sizeof(points_prefix)-1, points_prefix) == 0);
    if (line.size() > 0)
      return (line.compare(0, sizeof(points_prefix)-1, points_prefix) == 0);
    return false;
  }

  void Su2MeshParser::get_dim(const std::string &line, Mesh &mesh)
  {
    if (line.size() < sizeof(elems_prefix)) return;
    std::string number = line.substr (sizeof("NDIME="));
    std::istringstream ss(number);
    ss >> m_dims;
    mesh.set_dims(m_dims);
    std::cout << "parsed dimension " << m_dims << std::endl;
  }

  void Su2MeshParser::get_elems(const std::string &line)
  {
    if (line.size() < sizeof(elems_prefix)) return;
    std::string number = line.substr (sizeof(elems_prefix));
    std::istringstream ss(number);
    ss >> m_elems;
    std::cout << "parsed num of elements  " << m_elems << std::endl;
  }

  void Su2MeshParser::get_points(const std::string &line, Mesh &mesh)
  {
    if (line.size() < sizeof(points_prefix)) return;
    std::string number = line.substr (sizeof(points_prefix));
    std::istringstream ss(number);
    ss >> m_points;
    mesh.set_num_nodes(m_points);
    std::cout << "parsed num of points  " << m_points << std::endl;
  }

  // geometry_structure.cpp in SU2
  void Su2MeshParser::parse_elems(const std::string &line)
  {
    get_elems(line);
    if (m_elems == 0)  return;
    assert(!m_mesh_file.eof());
    int elem_idx = 0;
    //    Su2ELemType VTK_Type;

    //    unsigned int iElem_Bound = 0, iPoint = 0, ielem_div = 0, ielem = 0, *Local2Global = NULL, vnodes_edge[2], vnodes_triangle[3], vnodes_quad[4], vnodes_tetra[4], vnodes_hexa[8],
    unsigned int vnodes_triangle[3];


    std::string _line;
    while(!m_mesh_file.eof()) {
      if (elem_idx == m_elems) return;
      getline(m_mesh_file, _line);
      if (is_comment(_line)) continue;
      istringstream elem_line(_line);      
      int type;
      elem_line >> type;
      Su2ElemType VTK_Type = static_cast<Su2ElemType>(type);
      switch(VTK_Type) {
      case Su2ElemType::TRIANGLE:
        elem_line >> vnodes_triangle[0]; elem_line >> vnodes_triangle[1]; elem_line >> vnodes_triangle[2];
        //        elem[ielem] = new CTriangle(vnodes_triangle[0], vnodes_triangle[1], vnodes_triangle[2], 2);
        //        ielem_div++; ielem++; nelem_triangle++;
        ++elem_idx;
        break;
      default:
        //        std::cout << _line << std::endl;
        printf("VTK type %d index %d.\n", type, elem_idx);
        assert(0); // TODO
      }
    }
    assert(elem_idx == m_elems);
  }
  
  void Su2MeshParser::parse_points(const std::string &line, Mesh &mesh)
  {
    get_points(line, mesh);
    if (m_points == 0)  return;
    assert(!m_mesh_file.eof());
    int point_idx = 0;
    //    Su2ELemType VTK_Type;
    //    geometry.resize(m_points);
    std::string _line;
    while(!m_mesh_file.eof()) {
      if (point_idx == m_points) return;
      getline(m_mesh_file, _line);
      if (is_comment(_line)) continue;
      istringstream point_line(_line);
      double pos[3];

      if (m_dims == 2) {
        point_line >> pos[0];         point_line >> pos[1];
        float2 f_pos = make_float2(pos[0], pos[1]);
        mesh.set_node_pos(point_idx, f_pos);
      } else if (m_dims == 3) {
        point_line >> pos[0];         point_line >> pos[1];         point_line >> pos[2];
        float4 f_pos = make_float4(pos[0], pos[1], pos[2], 1.0f);
        mesh.set_node_pos(point_idx, f_pos);
      }
      //      printf("%f %f\n",pos[0], pos[1]);
      int index;
      point_line >> index;
      assert(index == point_idx);
      point_idx++;
    }
    assert(point_idx == m_points);
  }
  

} // namespace 
