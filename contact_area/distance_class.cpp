#include "distance_class.hpp"




// Constructor 
// string sheet_dump_file, string lb_dump_file
DistanceCalculator::DistanceCalculator(string sheet_dump, string lb_dump, string outname_input){
    string line;
    outname = outname_input;

    // --- Get info --- //
    // Sheet 
    sheet_infile.open(sheet_dump);
    readlines(sheet_infile, line, 2); timestep = stoi(line);
    readlines(sheet_infile, line, 2); sheet_num_atoms = stoi(line);
    readlines(sheet_infile, line, 5); // read remaining info
    // sheet_infile.close();

    // Lower block 
    lb_infile.open(lb_dump);
    readlines(lb_infile, line, 2); assert(timestep == stoi(line) && "timesteps do not match");
    readlines(lb_infile, line, 2); lb_num_atoms = stoi(line);
    readlines(lb_infile, line, 5); // read remaining info
    // lb_infile.close();

    // Allocate position and distance arrays
    distances = static_cast<double*>(malloc(sheet_num_atoms * sizeof(distances)));
    alloc2D(&sheet_atom_pos, sheet_num_atoms, 3);
    alloc2D(&lb_atom_pos, lb_num_atoms, 3);


    // Go back to first line for better workflow
    sheet_infile.seekg(0, ios::beg);
    lb_infile.seekg(0, ios::beg);
}



int DistanceCalculator::read_timestep(){
    string line;
    int id;

    // Check for EOF
    if (!getline(sheet_infile, line)){
        cout << "EOF" << endl;
        return 0;
    }



    // --- Get timestep and skip info lines --- //
    // Sheet
    readlines(sheet_infile, line, 1); timestep = stoi(line);
    readlines(sheet_infile, line, 7); // read remaining info

    // Lower block 
    readlines(lb_infile, line, 2); assert(timestep == stoi(line) && "timesteps do not match");
    readlines(lb_infile, line, 7); // read remaining info

   
    // --- Fill in position arrays --- //
    // Sheet 
    for (size_t i=0; i<sheet_num_atoms; i++){
        // id x y z
        sheet_infile >> id >> sheet_atom_pos[i][0] >> sheet_atom_pos[i][1] >> sheet_atom_pos[i][2]; //  >> vx >> vy >> vz;
       
        // cout << "id: " << id 
        // << ", pos: [" << sheet_atom_pos[i][0] 
        // << ", " << sheet_atom_pos[i][1] 
        // << ", " << sheet_atom_pos[i][2] 
        // << "]" << endl;
        
    }


    // Lower block
    for (size_t i=0; i<lb_num_atoms; i++){
        // id x y z
        lb_infile >> id >> lb_atom_pos[i][0] >> lb_atom_pos[i][1] >> lb_atom_pos[i][2]; //  >> vx >> vy >> vz;
        
        // cout << "id: " << id 
        // << ", pos: [" << lb_atom_pos[i][0] 
        // << ", " << lb_atom_pos[i][1] 
        // << ", " << lb_atom_pos[i][2] 
        // << "]" << endl;
        
    }

    // Add linebreaks
    readlines(sheet_infile, line, 1);
    readlines(lb_infile, line, 1);

    return 1;


}


void DistanceCalculator::calculate_minimum_distance(){
    double min_sqrdis, sqrdis;
    for (size_t i=0; i<sheet_num_atoms; i++){
        min_sqrdis = 1e5;
        for (size_t j=0; j< lb_num_atoms; j++){
            sqrdis = (sheet_atom_pos[i][0] - lb_atom_pos[j][0])*(sheet_atom_pos[i][0] - lb_atom_pos[j][0]) + 
                     (sheet_atom_pos[i][1] - lb_atom_pos[j][1])*(sheet_atom_pos[i][1] - lb_atom_pos[j][1]) + 
                     (sheet_atom_pos[i][2] - lb_atom_pos[j][2])*(sheet_atom_pos[i][2] - lb_atom_pos[j][2]);

            if (sqrdis <  min_sqrdis){
                min_sqrdis = sqrdis;
            }
        }
    
        distances[i] = sqrt(min_sqrdis);
    }
    
    // for (size_t i=0; i<sheet_num_atoms; i++){
    //     cout << distances[i] << endl;
    // }

}


void DistanceCalculator::write_distances(bool append){
    // Write to file 
    if (not append){
        cout << "Open the files" << endl;
    }

//  ofstream outflip;
//   string output_file = fileout + "_flipped.txt";
//   if (Temp == InitialTemp){
//     outflip.open(output_file, ios::out);}
//   else{
//     outflip.open(output_file, ios::out | ios::app);
//   }
//   outflip << setw(15) << setprecision(8) << Temp;
//   outflip << setw(15) << setprecision(8) << float(accepted_flips)/MCcycles/NSpins/NSpins << endl;
//   outflip.close();
// }

// void Functions::WriteEnergyState(double E, int cycles, int NSpins, double Temp, string fileout){
//   ofstream outstate;
//   string output_file = fileout + "_EnergyStates.txt";
//   if (cycles == 1){
//     outstate.open(output_file, ios::out);
//     outstate << "Temp= " << Temp << endl;
//     outstate << "NSpins= " << NSpins << endl;
//   }
//   else{
//     outstate.open(output_file, ios::out | ios::app);
//   }
//   outstate << setw(15) << setprecision(8) << cycles;
//   outstate << setw(15) << setprecision(8) << E << endl;
//   outstate.close();
// }


}


void DistanceCalculator::readlines(ifstream &infile, string &line, int N){
     for (size_t i=0; i<N; i++){getline(infile, line);}
}


void DistanceCalculator::alloc2D(double ***A, int m, int n){
    *A = static_cast<double**>(malloc(m * sizeof(*A)));
    (*A)[0] = static_cast<double*>(malloc(m*n * sizeof(*A)[0])); // Underlying 2D array

    for (size_t  i=1; i<m; i++){
        (*A)[i] = &((*A)[0][i*n]);
    }
}
