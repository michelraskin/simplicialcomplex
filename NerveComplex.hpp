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
    std::vector<std::pair<SimplicialComplex, double>> theSimplicialComplexes{};
    // MatrixXd theDistanceMatrixSorted;

    double theMaxDistance;
    double theMinDistance;

    struct BirthDeathStruct
    {
        VectorXd theVector;
        double theBirth;
        double theDeath;
    };

    std::array<vector<BirthDeathStruct>, 4> theBirhDeaths;

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

        for (size_t myDim = 0; myDim < 5; myDim++)
        {
            std::cout << "Getting birth death of dimesion " << myDim << std::endl;
            getBirthAndDeathRate(myDim);
        }
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
            theSimplicialComplexes.push_back(std::pair<SimplicialComplex, double>(mySimplicialComplex, myRadius));
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

            FullPivLU<MatrixXd> mySolver(myAlive);
            MatrixXd myLinearDependence = mySolver.solve(myHomology);
            MatrixXd myLinearDependenceKernel = kernel(myLinearDependence);
            myHomology = myHomology * myLinearDependenceKernel; // Can also just check one by one


            FullPivLU<MatrixXd> mySolver(myHomology);
            MatrixXd myLinearDependence = mySolver.solve(myAlive);
            MatrixXd myLinearDependenceKernel = kernel(myLinearDependence);
            myHomology = myAlive * myLinearDependenceKernel; // Will return all dead cycles 

            */  
            mySimplicialComplex.printEulerCharacteristic();
        }
    }

    void getBirthAndDeathRate(size_t aDim)
    {   
        vector<BirthDeathStruct> myBirthDeath{};
        MatrixXd myCurrentBoundary{};
        vector<Simplex> myCurrentIndex{};
        for (const auto& [mySimplicial, myRadius] : theSimplicialComplexes)
        {
            auto myCurrentHomology = mySimplicial.getHomology(aDim);
            if (myCurrentBoundary.size() == 0 || myCurrentIndex.size() == 0)
            {
                myCurrentBoundary = myCurrentHomology;
                myCurrentIndex = mySimplicial.getSimplicesPerDimOrdered(aDim+1);
                if (myCurrentIndex.size() > 0)
                {
                    for (long i = 0; i < myCurrentHomology.cols(); i++)
                    {
                        auto myBirthDeathSt = BirthDeathStruct{};
                        myBirthDeathSt.theVector = myCurrentHomology.col(i);
                        myBirthDeathSt.theBirth = myRadius;
                        myBirthDeathSt.theDeath = -1;
                        myBirthDeath.push_back(myBirthDeathSt);
                    }
                }
            }
            else
            {
                resizeBirthDeath(myBirthDeath, myCurrentBoundary, myCurrentIndex, mySimplicial.getSimplicesPerDimOrdered(aDim+1));
                MatrixXd myNewBoundary(myCurrentBoundary.rows(), 0);
                for (auto& myBd : myBirthDeath)
                {
                    if (myBd.theDeath == -1 && !isLinearlyIndependent(myCurrentHomology, myBd.theVector))
                    {
                        if (myNewBoundary.cols() == 0 || isLinearlyIndependent(myNewBoundary, myBd.theVector))
                        {
                            myNewBoundary.resize(myCurrentBoundary.rows(), myNewBoundary.cols() + 1);
                            myNewBoundary.col(myNewBoundary.cols()-1) = myBd.theVector;
                        }
                        else
                        {
                            myBd.theDeath = myRadius;
                        }
                    }
                    else if (myBd.theDeath == -1)
                    {
                        myBd.theDeath = myRadius;
                    }
                }
                myCurrentBoundary = myNewBoundary;
                for (long i = 0; i < myCurrentHomology.cols(); i++)
                {
                    if (isLinearlyIndependent(myCurrentBoundary, myCurrentHomology.col(i)))
                    {
                        auto myBirthDeathSt = BirthDeathStruct{};
                        myBirthDeathSt.theVector = myCurrentHomology.col(i);
                        myBirthDeathSt.theBirth = myRadius;
                        myBirthDeathSt.theDeath = -1;
                        if (!isLinearlyIndependent(myNewBoundary, myCurrentHomology.col(i)))
                        {
                            myBirthDeathSt.theDeath = myRadius;
                        }
                        else
                        {
                            myNewBoundary.resize(myCurrentBoundary.rows(), myNewBoundary.cols() + 1);
                            myNewBoundary.col(myNewBoundary.cols()-1) = myCurrentHomology.col(i);
                        }
                        myBirthDeath.push_back(myBirthDeathSt);
                    }
                }
                myCurrentBoundary = myNewBoundary;
            }
        }

        for (auto& myBd : myBirthDeath)
        {
            std::cout << "Vector born at: " << myBd.theBirth << std::endl;
            std::cout << "Vector deat at: " << myBd.theDeath << std::endl;
            std::cout << myBd.theVector << std::endl;
        }

        theBirhDeaths[aDim] = myBirthDeath;
    }

    void resizeBirthDeath(vector<BirthDeathStruct>& aBirthDeath, MatrixXd& aCurrentBoundary, vector<Simplex>& myCurrentIndex, const auto& myNewIndex)
    {
        size_t j = 0;
        for (size_t i = 0; i < myNewIndex.size(); i++)
        {
            if (myCurrentIndex[j] != myNewIndex[i])
            {
                for (auto& myBd : aBirthDeath)
                {
                    auto vec = myBd.theVector;
                    VectorXd newVec(vec.size() + 1);
                    if (j!= 0)
                        newVec.head(j) = vec.head(j);
                    newVec(j) = 0;
                    newVec.tail(vec.size() - j) = vec.tail(vec.size() - j);
                    myBd.theVector = newVec;
                }
            }
            else
            {
                j++;
            }
        }
        if (myNewIndex.size() > 0)
        {
            aCurrentBoundary.resize(myNewIndex.size(), aCurrentBoundary.cols());
            size_t i = 0;
            for (auto& myBd : aBirthDeath)
            {
                if (myBd.theDeath != -1)
                {
                    aCurrentBoundary.col(i) = myBd.theVector;
                }
            }
            myCurrentIndex = myNewIndex;
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