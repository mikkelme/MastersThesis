 #ifndef DISTANCE_CLASS_H
#define DISTANCE_CLASS_H


#include <iostream>   // input and output
#include <fstream>    // ofstream
#include <cstdlib>    // atof function

using namespace std;

// #include <iostream>   // input and output
// using namespace std;


class DistanceCalculator
{
public:
  DistanceCalculator(string sheet_dump, string lb_dump); //Constructor
  int read_timestep();
  void readlines(ifstream &infile, string &line, int N);
  void calculate_minimum_distance();

  void alloc2D(double ***A, int m, int n);
  int timestep, sheet_num_atoms, lb_num_atoms; 
  // string sheet_dump, lb_dump;
  ifstream sheet_infile, lb_infile;

  double *distances;
  double **sheet_atom_pos, **lb_atom_pos;








};



#endif //DISTANCE_CLASS_H