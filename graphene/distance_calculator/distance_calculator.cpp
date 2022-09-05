#include "distance_class.hpp"








int main(int argc, char* argv[]){


    // Initialize instance of distance_calculator     

    string sheet_dump = "../sheet.data"; 
    string lb_dump = "../sheet.data";  // lover block


    DistanceCalculator my_calc(sheet_dump, lb_dump);










} // end of main 


// Compile as: g++ distance_calculator.cpp distance_class.cpp -o dis_calc.out
