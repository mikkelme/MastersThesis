#include "distance_class.hpp"






int main(int argc, char* argv[]){
    string sheet_dump, sub_dump, outname;
    clock_t t1, t2;
    double duration_seconds;

    // Filenames 
    sheet_dump = "../area_vs_stretch/sheet.data";
    sub_dump = "../area_vs_stretch/substrate_contact.data";
    outname = "distances.txt";

    // Update with command line arguments if present
    if (argc > 1){sheet_dump = argv[1];}
    if (argc > 2){sub_dump = argv[2];}
    if (argc > 3){outname = argv[3];}



    // cout << sheet_dump << endl;
    // cout << sub_dump << endl;
    // cout << outname << endl;
    // exit(0);

    // Initialize instance of distance calculator class     
    DistanceCalculator dis_calc(sheet_dump, sub_dump, outname);
    cout << "--> Processing minimum distance" << std::endl;

    // Start measuring time
    t1 = clock();  

    // Read -> calculate min distance -> write to file
    while (dis_calc.read_timestep()){
        cout << '\r' << "Timestep: " << dis_calc.timestep << flush;
        dis_calc.calculate_minimum_distance();
        dis_calc.write_distances();
    }
    // Stop measuring time
    t2 = clock();

    // Calculate the elapsed time.
    duration_seconds = ((double) (t2 - t1)) / CLOCKS_PER_SEC;
    cout << "--> Done: Time used = " << duration_seconds <<  " s" << endl;


    // Close files
    dis_calc.sheet_infile.close();
    dis_calc.sub_infile.close();
    dis_calc.ofile.close();





} // end of main 

// Compile as: g++ distance_calculator.cpp distance_class.cpp -o dis_calc.out
