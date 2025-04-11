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
    array<vector<Simplex>, 5> theIndex;
    std::vector<std::pair<SimplicialComplex, double>> theSimplicialComplexes{};
    std::string theName;
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
        theName = aFile;
        theMatrix = readCsv(aFile);
        theDistanceMatrix = calculateDistances();
        theMaxDistance = theDistanceMatrix.col(2).maxCoeff();
        theMinDistance = theDistanceMatrix.col(2).minCoeff();
        doFiltration(0.025, theMaxDistance / 5);

        for (size_t myDim = 0; myDim < 4; myDim++)
        {
            std::cout << "Getting birth death of dimesion " << myDim << std::endl;
            getBirthAndDeathRate(myDim);
        }
        for (size_t myDim = 0; myDim < 4; myDim++)
        {
            std::cout << theBirhDeaths[myDim].size() << std::endl;
            writeCSV(theName + "birthDeath" + std::to_string(myDim) + ".csv", theBirhDeaths[myDim], myDim);
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
            if (mySimplices.size() > 1200)
            {
                // Prevent explosion in computations
                break;
            }
            auto mySimplicialComplex = SimplicialComplex{mySimplices};
            // mySimplicialComplex.printComplex();
            for (size_t i = 1; i < 5; i++)
            {
                std::cout << "Printing Boundary of dimension " << i -1 << std::endl;
                mySimplicialComplex.printHomology(i, false);
            }
            theSimplicialComplexes.push_back(std::pair<SimplicialComplex, double>(mySimplicialComplex, myRadius));
            mySimplicialComplex.printEulerCharacteristic();
        }
    }

    void getBirthAndDeathRate(size_t aDim)
    {   
        vector<BirthDeathStruct> myBirthDeath{};
        MatrixXd myCurrentBoundary{};
        vector<Simplex> myCurrentIndex{};
        bool myInitialized = false;
        for (const auto& [mySimplicial, myRadius] : theSimplicialComplexes)
        {
            auto myCurrentHomology = mySimplicial.getHomology(aDim);
            if ((myCurrentBoundary.size() == 0 || (getRank(myCurrentBoundary) == 0) || myCurrentIndex.size() == 0) && !myInitialized)
            {
                myCurrentIndex = mySimplicial.getSimplicesPerDimOrdered(aDim+1);
                if (myCurrentIndex.size() > 0 && (getRank(myCurrentHomology) > 0))
                {
                    myCurrentBoundary = myCurrentHomology;
                    for (long i = 0; i < myCurrentHomology.cols(); i++)
                    {
                        auto myBirthDeathSt = BirthDeathStruct{};
                        myBirthDeathSt.theVector = myCurrentHomology.col(i);
                        myBirthDeathSt.theBirth = myRadius;
                        myBirthDeathSt.theDeath = -1;
                        myBirthDeath.push_back(myBirthDeathSt);
                    }
                    myInitialized = true;
                }
            }
            else
            {
                resizeBirthDeath(myBirthDeath, myCurrentBoundary, myCurrentIndex, mySimplicial.getSimplicesPerDimOrdered(aDim+1));
                MatrixXd myNewBoundary(myCurrentBoundary.rows(), 0);
                for (auto& myBd : myBirthDeath)
                {
                    // Checking for cycles that will die
                    if (myBd.theDeath == -1 && !isLinearlyIndependent(myCurrentHomology, myBd.theVector))
                    {
                        if (myNewBoundary.cols() == 0 || isLinearlyIndependent(myNewBoundary, myBd.theVector))
                        {
                            MatrixXd myNewBoundary2(myCurrentBoundary.rows(), myNewBoundary.cols() + 1);
                            myNewBoundary2.setZero();
                            for (long f = 0; f < myNewBoundary.cols(); f++)
                            {
                                myNewBoundary2.col(f) = myNewBoundary.col(f);
                            }
                            myNewBoundary = myNewBoundary2;
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
                    // Checking new cycles
                    if (myCurrentBoundary.size() == 0 || isLinearlyIndependent(myCurrentBoundary, myCurrentHomology.col(i)))
                    {
                        auto myBirthDeathSt = BirthDeathStruct{};
                        myBirthDeathSt.theVector = myCurrentHomology.col(i);
                        myBirthDeathSt.theBirth = myRadius;
                        myBirthDeathSt.theDeath = -1;
                        if (myNewBoundary.size() != 0 && !isLinearlyIndependent(myNewBoundary, myCurrentHomology.col(i)))
                        {
                            myBirthDeathSt.theDeath = myRadius;
                        }
                        else
                        {
                            MatrixXd myNewBoundary2(myCurrentBoundary.rows(), myNewBoundary.cols() + 1);
                            myNewBoundary2.setZero();
                            for (long f = 0; f < myNewBoundary.cols(); f++)
                            {
                                myNewBoundary2.col(f) = myNewBoundary.col(f);
                            }
                            myNewBoundary = myNewBoundary2;
                            myNewBoundary.col(myNewBoundary.cols()-1) = myCurrentHomology.col(i);
                        }
                        myBirthDeath.push_back(myBirthDeathSt);
                    }
                }
                myCurrentBoundary = myNewBoundary;
            }
        }

        // for (auto& myBd : myBirthDeath)
        // {
        //     std::cout << "Vector born at: " << myBd.theBirth << std::endl;
        //     std::cout << "Vector deat at: " << myBd.theDeath << std::endl;
        //     std::cout << myBd.theVector << std::endl;
        // }

        theBirhDeaths[aDim] = myBirthDeath;
        theIndex[aDim] = myCurrentIndex;
    }

    void resizeBirthDeath(vector<BirthDeathStruct>& aBirthDeath, MatrixXd& aCurrentBoundary, vector<Simplex>& myCurrentIndex, const auto& myNewIndex)
    {
        size_t j = 0;
        for (size_t i = 0; i < myNewIndex.size(); i++)
        {
            if (j >= myCurrentIndex.size())
            {
                for (auto& myBd : aBirthDeath)
                {
                    auto vec = myBd.theVector;
                    VectorXd newVec = vec;
                    newVec.resize(vec.size() + 1);
                    if (j!= 0)
                        newVec.head(j) = vec.head(j);
                    newVec(newVec.size()-1) = 0;
                    myBd.theVector = newVec;
                }
            }
            else if (myCurrentIndex[j] != myNewIndex[i])
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
                if (myBd.theDeath == -1)
                {
                    aCurrentBoundary.col(i) = myBd.theVector;
                    i++;
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
                        if ((theDistanceMatrix(myIndex - 1, i) > 2 * aRadius))
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

    void writeCSV(const std::string& aFileName, const vector<BirthDeathStruct>& aVec, size_t aDim) 
    {
        std::ofstream myFile(aFileName);
        if (myFile.is_open()) 
        {
            for (const auto& aBirthDeath : aVec)
            {
                // In order to print nicely I need to save the index too, the only problem is that this won't be csv anymore
                myFile << aBirthDeath.theBirth << "," << aBirthDeath.theDeath << "," ;
                for (long i = 0; i < aBirthDeath.theVector.rows(); i++)
                {
                    if (std::round(aBirthDeath.theVector(i)) == 1)
                    {
                        myFile << " + " << std::to_string((int)std::round(aBirthDeath.theVector(i))) << " * " << theIndex[aDim][i];
                    }

                    if (std::round(aBirthDeath.theVector(i)) == -1)
                    {
                        myFile << std::to_string((int)std::round(aBirthDeath.theVector(i))) << " * " << theIndex[aDim][i];
                    }
                }
                myFile << "\n";
            }
            myFile.close();
            std::cout << "Saved to " << aFileName << "\n";
        } else {
            std::cerr << "Could not open file for writing.\n";
        }
    }
};