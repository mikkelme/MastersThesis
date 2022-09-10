#ifndef DISTANCE_CLASS_H
#define DISTANCE_CLASS_H


#include <iostream>   // input and output
#include <fstream>    // ofstream
#include <cstdlib>    // atof function
#include <time.h>     // Timing

using namespace std;

// #include <iostream>   // input and output
// using namespace std;


class DistanceCalculator
{
public:
  DistanceCalculator(string sheet_dump, string sub_dump, string outname_input); //Constructor
  int read_timestep();
  void readlines(ifstream &infile, string &line, int N);
  void calculate_minimum_distance();
  void write_distances();

  void alloc2D(double ***A, int m, int n);
  int timestep, sheet_num_atoms, sub_num_atoms; 
  string outname;
  ifstream sheet_infile, sub_infile;
  ofstream ofile;


  double *distances;
  double **sheet_atom_pos, **sub_atom_pos;








};



#endif //DISTANCE_CLASS_H