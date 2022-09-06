#include "distance_class.hpp"








int main(int argc, char* argv[]){

    // Filenames 
    string sheet_dump = "../sheet.data"; 
    string lb_dump = "../lower_block.data";  // lower block


    // Initialize instance of distance calculator class     
    DistanceCalculator dis_calc(sheet_dump, lb_dump);



    while (dis_calc.read_timestep()){
        cout << dis_calc.timestep << endl;
        dis_calc.calculate_minimum_distance();
        // dis_cal.write_distances();
        // exit(0);

    }





    dis_calc.sheet_infile.close();
    dis_calc.lb_infile.close();







} // end of main 


// Compile as: g++ distance_calculator.cpp distance_class.cpp -o dis_calc.out
