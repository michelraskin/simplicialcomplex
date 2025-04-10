#include <iostream>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <string>
#include <Eigen/Dense>
#include "Simplex.hpp"
#include "SimplexUtils.hpp"

class NerveComplex {
private:
    MatrixXd theMatrix;
    MatrixXd theDistanceMatrix;
    // MatrixXd theDistanceMatrixSorted;

    double theMaxDistance;
    double theMinDistance;
public:
    NerveComplex(string aFile) 
    {
        theMatrix = readCsv(aFile);
        theDistanceMatrix = calculateDistances();
        // theDistanceMatrixSorted = theDistanceMatrix;
        // sortRowsByColumn(theDistanceMatrixSorted, 2);
        theMaxDistance = theDistanceMatrix.col(2).maxCoeff();
        theMinDistance = theDistanceMatrix.col(2).minCoeff();
        doFiltration(0.1, theMaxDistance / 3 + 1);
    }

    MatrixXd calculateDistances()
    {
        MatrixXd myDistanceMatrix(theMatrix.rows(), theMatrix.rows());
        for (long i = 0; i < theMatrix.rows(); i++)
        {
            VectorXd myVec1 = theMatrix.row(i);
            myVec1(0, 0) = 0;
            for (long j = 0; j < theMatrix.rows(); j++)
            {
                VectorXd myVec2 = theMatrix.row(j);
                myVec2(0, 0) = 0;
                double myDistance = getEuclideanDistance(myVec1, myVec2);
                myDistanceMatrix(i, j) = myDistance;
            }
        }
        return myDistanceMatrix;
    }

    void doFiltration(double aRadius, double aMaxRadius)
    {
        for (double myRadius = 0; myRadius < aMaxRadius + aRadius; myRadius+=aRadius)
        {
            std::cout << " Radius "  << myRadius << std::endl;
            unordered_set<Simplex> mySimplices{};
            auto myNewSimplex = getOneDimensionalSimplex(mySimplices);
            std::cout << "Computing edges" << std::endl;
            myNewSimplex = getNDimensionalSimplex(mySimplices, myNewSimplex, myRadius);
            std::cout << "Computing Triangles" << std::endl;
            myNewSimplex = getNDimensionalSimplex(mySimplices, myNewSimplex, myRadius);
            std::cout << "Computing Tetrahedra" << std::endl;
            myNewSimplex = getNDimensionalSimplex(mySimplices, myNewSimplex, myRadius);
            std::cout << "Computing Simplicial" << std::endl;
            auto mySimplicialComplex = SimplicialComplex{mySimplices};
            // mySimplicialComplex.printComplex();
            for (size_t i = 1; i < 5; i++)
            {
                std::cout << "Printing Boundary of dimension " << i -1 << std::endl;
                mySimplicialComplex.printHomology(i, i==2);
            }
            /*
            TODO:
            1. For each column in the homology function record when it's born (essentially when a cycle is born)
            2. A cycle is born if it's not a linear combination of alive cycles at that step
            3. Each step in order to do this check the dimensions (number of rows) need to be expanded to match
            4. A cycle is dead when it's not a part of the current homologies

            Implementation:
            1. Vector that contains the born columns, the age where it was born and the age where it died
            2. Each time something is born we push into the vector
            3. When something dies we don't remove just keep and don't use to check against new born
            4. Can check for linear dependence the same way I do for the kernel
            */  
            mySimplicialComplex.printEulerCharacteristic();
        }
    }

    unordered_set<Simplex> getOneDimensionalSimplex(unordered_set<Simplex>& aSimplices)
    {
        // One dimensional
        for (long i = 0; i < theMatrix.rows(); i++)
        {
            auto mySimplexString = vector<string>{std::to_string((int)std::round(theMatrix(i, 0)))};
            aSimplices.insert(Simplex(mySimplexString));
        }
        return aSimplices;
    }

    unordered_set<Simplex> getNDimensionalSimplex(unordered_set<Simplex>& aSimplices, unordered_set<Simplex>& aSimplicesNMinus1D, double aRadius)
    {
        // Two dimensional
        auto mySimplicesNew = unordered_set<Simplex>{};
        for (const auto& mySimplex : aSimplicesNMinus1D)
        {
            for (long i = 0; i < theMatrix.rows(); i++)
            {
                auto mySimplexString = vector<string>{std::to_string((int)std::round(theMatrix(i, 0)))};
                if (!mySimplex.contains(Simplex(mySimplexString)))
                {
                    bool aConditionMet = true;
                    for (const auto& myIndexStr : mySimplex.getOrientedSimplex())
                    {
                        int myIndex = std::stoi(myIndexStr);
                        if ((theDistanceMatrix(myIndex - 1, i) > aRadius))
                        {
                            aConditionMet = false;
                            break;
                        }
                    }
                    if (aConditionMet)
                    {
                        auto myNewSimplex = mySimplex.getOrientedSimplex();
                        myNewSimplex.push_back(mySimplexString.back());
                        aSimplices.insert(myNewSimplex);
                        mySimplicesNew.insert(myNewSimplex);
                    }
                }
            }
        }
        return mySimplicesNew;
    }
};