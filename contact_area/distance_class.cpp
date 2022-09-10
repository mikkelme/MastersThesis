#include "distance_class.hpp"




// Constructor 
// string sheet_dump_file, string sub_dump_file
DistanceCalculator::DistanceCalculator(string sheet_dump, string sub_dump, string outname_input){
    string line;
    outname = outname_input;

    // --- Get info --- //
    // Sheet 
    sheet_infile.open(sheet_dump);
    if (!sheet_infile){ // Check for existence
        cout << "FILE NOT FOUND: " << sheet_dump <<  endl; 
        exit(1);
    }

    readlines(sheet_infile, line, 2); timestep = stoi(line);
    readlines(sheet_infile, line, 2); sheet_num_atoms = stoi(line);
    readlines(sheet_infile, line, 5); // read remaining info

    // Substrate 
    sub_infile.open(sub_dump);
    if (!sub_infile){  // Check for existence
        cout << "FILE NOT FOUND: " << sub_dump <<  endl; 
        exit(1);
    }

    readlines(sub_infile, line, 2); assert(timestep == stoi(line) && "timesteps do not match");
    readlines(sub_infile, line, 2); sub_num_atoms = stoi(line);
    readlines(sub_infile, line, 5); // read remaining info

    // Allocate position and distance arrays
    distances = static_cast<double*>(malloc(sheet_num_atoms * sizeof(distances)));
    alloc2D(&sheet_atom_pos, sheet_num_atoms, 3);
    alloc2D(&sub_atom_pos, sub_num_atoms, 3);


    // Go back to first line for better workflow
    sheet_infile.seekg(0, ios::beg);
    sub_infile.seekg(0, ios::beg);

    // Open outfile as well
    ofile.open(outname);

}



int DistanceCalculator::read_timestep(){
    string line;
    int id;

    // Check for EOF
    if (!getline(sheet_infile, line)){
        cout << " (EOF) " << endl; // EOF
        return 0;
    }


    // --- Get timestep and skip info lines --- //
    // Sheet
    readlines(sheet_infile, line, 1); timestep = stoi(line);
    readlines(sheet_infile, line, 7); // read remaining info

    // Substarte
    readlines(sub_infile, line, 2); assert(timestep == stoi(line) && "timesteps do not match");
    
    readlines(sub_infile, line, 7); // read remaining info

   
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


    // Substrate
    for (size_t i=0; i<sub_num_atoms; i++){
        // id x y z
        sub_infile >> id >> sub_atom_pos[i][0] >> sub_atom_pos[i][1] >> sub_atom_pos[i][2]; //  >> vx >> vy >> vz;
        
        // cout << "id: " << id 
        // << ", pos: [" << sub_atom_pos[i][0] 
        // << ", " << sub_atom_pos[i][1] 
        // << ", " << sub_atom_pos[i][2] 
        // << "]" << endl;
        
    }

    // Add linebreaks
    readlines(sheet_infile, line, 1);
    readlines(sub_infile, line, 1);

    return 1;


}


void DistanceCalculator::calculate_minimum_distance(){
    double min_sqrdis, sqrdis;
    for (size_t i=0; i<sheet_num_atoms; i++){
        min_sqrdis = 1e5;
        for (size_t j=0; j< sub_num_atoms; j++){
            sqrdis = (sheet_atom_pos[i][0] - sub_atom_pos[j][0])*(sheet_atom_pos[i][0] - sub_atom_pos[j][0]) + 
                     (sheet_atom_pos[i][1] - sub_atom_pos[j][1])*(sheet_atom_pos[i][1] - sub_atom_pos[j][1]) + 
                     (sheet_atom_pos[i][2] - sub_atom_pos[j][2])*(sheet_atom_pos[i][2] - sub_atom_pos[j][2]);

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


void DistanceCalculator::write_distances(){
    int width = 17;
    int prec = 8;

    // Append to file 
    ofile << "Timestep" << endl;
    ofile << timestep << endl;
    ofile << "Num sheet atoms" << endl;
    ofile << sheet_num_atoms << endl;
    ofile << "Sheet atom idx, min distance " << endl;
    for (size_t i = 0; i < sheet_num_atoms; i++)
    {
        ofile << setw(width) << setprecision(prec) << scientific << i;
        ofile << setw(width) << setprecision(prec) << scientific << distances[i] << endl;

    }
    


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
