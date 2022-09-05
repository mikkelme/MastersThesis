#include "distance_class.hpp"




// Constructor 
// string sheet_dump_file, string lb_dump_file
DistanceCalculator::DistanceCalculator(string sheet_dump, string lb_dump){
    string line;

    // --- Get info --- //
    // Sheet 
    ifstream sheet_infile;
    sheet_infile.open(sheet_dump);
    readlines(sheet_infile, line, 2); timestep = stoi(line);
    readlines(sheet_infile, line, 2); sheet_num_atoms = stoi(line);
    readlines(sheet_infile, line, 5); // read remainging info
    sheet_infile.close();


    // Lower block 
    ifstream lb_infile;
    lb_infile.open(lb_dump);
    readlines(lb_infile, line, 2); assert(timestep == stoi(line) && "timesteps do not match");
    readlines(lb_infile, line, 2); lb_num_atoms = stoi(line);
    readlines(lb_infile, line, 5); // read remainging info
    lb_infile.close();

    // Allocate position arrays
    alloc2D(&sheet_atom_pos, sheet_num_atoms, 3);
    alloc2D(&lb_atom_pos, lb_num_atoms, 3);

    // Next step read a timestep at a time and feel in these bad boys.....

}

void DistanceCalculator::readlines(ifstream &infile, string &line, int N){
     for (int i=0; i<N; i++){getline(infile, line);}
}



void DistanceCalculator::alloc2D(double ***A, int m, int n){
    *A = static_cast<double**>(malloc(m * sizeof(*A)));
    (*A)[0] = static_cast<double*>(malloc(m*n * sizeof(*A)[0])); // Underlying 2D array

    for (size_t  i=1; i<m; i++){
        (*A)[i] = &((*A)[0][i*n]);
    }
}
