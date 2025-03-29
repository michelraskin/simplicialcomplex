#include <iostream>
#include <set>
#include <vector>
#include <iomanip>
#include <string>
#include "Simplex.hpp"
#include "SimplicialComplex.hpp"

using namespace std;

int main() {
    vector<vector<string>> mySimplicesA = {
        {"Cow", "Rabbit"}, {"Cow", "Horse"}, {"Cow", "Dog"}, {"Rabbit", "Horse"}, {"Rabbit", "Dog"}, {"Horse", "Dog"},
        {"Fish", "Dolphin"}, {"Fish", "Oyster"}, {"Dolphin", "Oyster"},
        {"Broccoli", "Fern"}, {"Broccoli", "Onion"}, {"Broccoli", "Apple"}, {"Fern", "Onion"}, {"Fern", "Apple"},
        {"Onion", "Apple"},
        {"Cow", "Rabbit", "Horse"}, {"Cow", "Rabbit", "Dog"}, {"Cow", "Horse", "Dog"}, {"Rabbit", "Horse", "Dog"},
        {"Fish", "Dolphin", "Oyster"},
        {"Broccoli", "Fern", "Onion"}, {"Broccoli", "Fern", "Apple"}, {"Broccoli", "Onion", "Apple"},
        {"Fern", "Onion", "Apple"}
    };

    vector<vector<string>> mySimplicesB = {
        {"Cow", "Rabbit"}, {"Cow", "Fish"}, {"Cow", "Oyster"}, {"Cow", "Broccoli"}, {"Cow", "Onion"},
        {"Cow", "Apple"}, {"Rabbit", "Fish"}, {"Rabbit", "Oyster"}, {"Rabbit", "Broccoli"}, {"Rabbit", "Onion"},
        {"Rabbit", "Apple"}, {"Fish", "Oyster"}, {"Fish", "Broccoli"}, {"Fish", "Onion"}, {"Fish", "Apple"},
        {"Oyster", "Broccoli"}, {"Oyster", "Onion"}, {"Oyster", "Apple"}, {"Broccoli", "Onion"}, {"Broccoli", "Apple"},
        {"Onion", "Apple"},
        {"Horse", "Dog"}, {"Horse", "Dolphin"}, {"Horse", "Fern"}, {"Dog", "Dolphin"}, {"Dog", "Fern"},
        {"Dolphin", "Fern"},
        {"Cow", "Broccoli", "Apple"}, {"Cow", "Onion", "Apple"}, {"Rabbit", "Broccoli", "Apple"},
        {"Rabbit", "Onion", "Apple"}, {"Fish", "Broccoli", "Apple"}, {"Fish", "Onion", "Apple"},
        {"Oyster", "Broccoli", "Apple"}, {"Oyster", "Onion", "Apple"}
    };

    vector<vector<string>> mySimplicesC = {
        {"A", "B"},{"A", "C"}, {"A", "D"},{"B", "E"},{"C", "E"},{"D", "E"}
    };

    cout << "===== Simplicial Complex A =====" << endl;
    SimplicialComplex myComplexA(mySimplicesA);
    myComplexA.printComplex();
    for (size_t i = 1; i < 4; i++)
    {
        std::cout << "Printing Boundary del " << i -1 << std::endl;
        myComplexA.printBoundaryMatrix(i);
        myComplexA.printBoundaryComplex(i);
        myComplexA.printBoundaryImage(i);
        myComplexA.printBoundaryKernel(i);
        myComplexA.printHomology(i);
    }
    cout << endl;

    cout << "===== Simplicial Complex B =====" << endl;
    SimplicialComplex myComplexB(mySimplicesB);
    myComplexB.printComplex();
    for (size_t i = 1; i < 4; i++)
    {
        std::cout << "Printing Boundary of dimension " << i -1 << std::endl;
        myComplexB.printBoundaryMatrix(i);
        myComplexB.printBoundaryComplex(i);
        myComplexB.printBoundaryImage(i);
        myComplexB.printBoundaryKernel(i);
        myComplexB.printHomology(i);
    }
    cout << endl;

    cout << "===== Simplicial Complex C =====" << endl;
    SimplicialComplex myComplexC(mySimplicesC);
    myComplexC.printComplex();
    for (size_t i = 1; i < 3; i++)
    {
        std::cout << "Printing Boundary of dimension " << i -1 << std::endl;
        myComplexC.printBoundaryMatrix(i);
        myComplexC.printBoundaryComplex(i);
        myComplexC.printBoundaryImage(i);
        myComplexC.printBoundaryKernel(i);
        myComplexC.printHomology(i);
    }
    cout << endl;

    return 0;
}