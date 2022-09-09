#include "distance_class.hpp"








int main(int argc, char* argv[]){

    // Filenames 
    string sheet_dump = "../sheet_dummy.data"; 
    string lb_dump = "../substrate_dummy.data";  
    string outname = "distances.txt";


    exit(1); // Working here <------------------
    // Initialize instance of distance calculator class     
    DistanceCalculator dis_calc(sheet_dump, lb_dump, outname);


    while (dis_calc.read_timestep()){
        cout << dis_calc.timestep << endl;
        // dis_calc.calculate_minimum_distance();
        // dis_cal.write_distances();
        // exit(0);
    }





    dis_calc.sheet_infile.close();
    dis_calc.lb_infile.close();







} // end of main 


// Compile as: g++ distance_calculator.cpp distance_class.cpp -o dis_calc.out
