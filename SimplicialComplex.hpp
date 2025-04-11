#pragma once
#include <iostream>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <string>
#include <Eigen/Dense>
#include <algorithm>
#include "Simplex.hpp"
#include "SimplexUtils.hpp"

using namespace std;
using namespace Eigen;

class SimplicialComplex {
private:
    static constexpr size_t MaxDimension = 7;
    unordered_set<Simplex> theSimplices;

    std::array<unordered_set<Simplex>, MaxDimension> theSimplicesPerDim;
    std::array<vector<Simplex>, MaxDimension> theSimplicesPerDimOrdered;
    std::array<MatrixXd, MaxDimension> theBoundaries{};
    std::array<MatrixXd, MaxDimension> theBoundariesUnoriented{};

    std::array<MatrixXd, MaxDimension> theHomologies{};

    vector<unordered_set<string>> theConnectedComponents;

public:
    SimplicialComplex(unordered_set<Simplex> aSimplices) 
    : SimplicialComplex([&aSimplices]() {
        vector<vector<string>> mySimplices{};
        for (const auto& mySimplex : aSimplices)
        {
            mySimplices.push_back(mySimplex.getOrientedSimplex());
        }
        std::stable_sort(mySimplices.begin(), mySimplices.end());
        return mySimplices;
      }())
      {}

    SimplicialComplex(vector<vector<string>> aSimplices) 
    {
        for (auto myStringVec : aSimplices) 
        {
            auto mySimplex = Simplex(myStringVec);
            theSimplices.insert(mySimplex);

            for (auto myValue : mySimplex.getUnorientedSimplex())
            {
                theSimplices.insert(Simplex{vector<string>{myValue}});

                theSimplicesPerDim[1].insert(Simplex{vector<string>{myValue}});
            }
            theSimplicesPerDim[mySimplex.size()].insert(mySimplex);

            vector<unordered_set<string>> myNewConnectedComponents{};
            unordered_set<int> mySetsWithComplex{};
            for (const auto &myValue : mySimplex.getUnorientedSimplex()) 
            {
                for (size_t i = 0; i < theConnectedComponents.size(); i++)
                {
                    if (theConnectedComponents[i].contains(myValue))
                    {
                        mySetsWithComplex.insert(i);
                    }
                }
            }

            unordered_set<string> myMergedSet{};
            myMergedSet.insert(mySimplex.getOrientedSimplex().begin(), mySimplex.getOrientedSimplex().end());
            for (auto mySetsToMerge : mySetsWithComplex)
            {
                myMergedSet.insert(theConnectedComponents[mySetsToMerge].begin(), theConnectedComponents[mySetsToMerge].end());
            }

            for (size_t i = 0; i < theConnectedComponents.size(); i++)
            {
                if (!mySetsWithComplex.contains(i))
                {
                    myNewConnectedComponents.push_back(theConnectedComponents[i]);
                }
            }
            myNewConnectedComponents.push_back(myMergedSet);

            theConnectedComponents = myNewConnectedComponents;
        }
        for (size_t i = 0; i < MaxDimension; i++)
        {
            for (auto myValue : theSimplicesPerDim[i])
            {
                theSimplicesPerDimOrdered[i].push_back(myValue);
            }
            std::stable_sort(theSimplicesPerDimOrdered[i].begin(), theSimplicesPerDimOrdered[i].end());
        }
        for (size_t i = 0; i < MaxDimension; i++)
        {
            std::cout << "Simplex of dim " << i << " has size " << theSimplicesPerDim[i].size() << std::endl;
        }
        std::cout << "Number of connected components " << theConnectedComponents.size() << std::endl;
        for (size_t i = 0; i < MaxDimension; i++)
        {
            auto now = std::chrono::system_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(now);
            std::cout << "[" << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << "] "
                    << "Computing oriented boundary " << i << std::endl;

            computeBoundaryMatrix<true>(i);

            // // Print time again
            // now = std::chrono::system_clock::now();
            // now_c = std::chrono::system_clock::to_time_t(now);
            // std::cout << "[" << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << "] "
            //         << "Computing unoriented boundary " << i << std::endl;

            // computeBoundaryMatrix<false>(i);
        }
    }

    const auto& getHomology(size_t aDim) const
    {
        return theHomologies[aDim];
    }

    const auto& getSimplicesPerDimOrdered(size_t aDim) const
    {
        return theSimplicesPerDimOrdered[aDim];
    }

    template<bool aOriented = false>
    void computeBoundaryMatrix(size_t aDim)
    {
        // Can refactor here by getting subsimplices
        auto& myBoundary = aOriented ? theBoundaries[aDim] : theBoundariesUnoriented[aDim];
        if (theSimplicesPerDimOrdered[aDim].size() == 0)
        {
            myBoundary.resize(1, 1);
            myBoundary(0, 0) = 0;
            return;
        }
        auto myRows = (theSimplicesPerDimOrdered[aDim-1].size() == 0) ? 1 : theSimplicesPerDimOrdered[aDim-1].size();
        auto myColumns = theSimplicesPerDimOrdered[aDim].size();
        myBoundary.resize(myRows, myColumns);
        myBoundary.setZero(myRows, myColumns);
        if (theSimplicesPerDimOrdered[aDim-1].size() == 0)
        {
            return;
        }
        size_t j = 0;
        for (auto mySimplexBase : theSimplicesPerDimOrdered[aDim])
        {
            const auto& mySubSimplices = mySimplexBase.getSubSimplices();
            for (const auto& mySubSimplex : mySubSimplices)
            {
                auto it = std::lower_bound(theSimplicesPerDimOrdered[aDim-1].begin(), theSimplicesPerDimOrdered[aDim-1].end(), mySubSimplex);
                int i = it - theSimplicesPerDimOrdered[aDim-1].begin();
                if constexpr (aOriented)
                {
                    myBoundary(i,j) = (mySubSimplex.getOrientation());
                }
                else
                {
                    myBoundary(i,j) = (1);
                }
            }
            j++;
        }
    }

    void printFunction(auto aThingsToPrint, bool aEndl = true)
    {
        for (const auto &myConnectedComponent : aThingsToPrint) {
            cout << "{";
            for (const auto& myValue : myConnectedComponent) {
                cout << setw(8) << myValue << ", ";
            }
            cout << "}";
            if (aEndl)
            {
                cout << endl;
            }
        }
    }

    void printComplex() 
    {
        cout << "Simplicial Complex (set of all simplices): " << endl;
        // printFunction(theSimplices);
        // cout << endl;

        cout << "Vertices: " << endl;
        printFunction(theSimplicesPerDimOrdered[1]);
        cout << endl;

        cout << "Edges: " << endl;
        printFunction(theSimplicesPerDimOrdered[2]);
        cout << endl;

        cout << "Triangles: " << endl;
        printFunction(theSimplicesPerDimOrdered[3]);
        cout << endl;

        cout << "Tetrahedra: " << endl;
        printFunction(theSimplicesPerDimOrdered[4]);
        cout << endl;

        cout << "Connected Components: " << endl;
        printFunction(theConnectedComponents);
        cout << endl;
    }

    void printBoundaryMatrix(size_t aDim)
    {
        cout << endl;
        cout << "A input vector shape: " << endl;
        printFunction(theSimplicesPerDimOrdered[aDim], true);
        cout << endl;
        for (long i = 0; i < theBoundaries[aDim].rows(); i++)
        {
            if (theSimplicesPerDimOrdered[aDim-1].size() > 0)
            {
                printFunction(unordered_set<Simplex>{theSimplicesPerDimOrdered[aDim-1][i]}, false); 
            }
            std::cout << "  [";
            for (long j = 0; j < theBoundaries[aDim].cols(); j++)
            {
                std::cout << setw(4) << (int)theBoundaries[aDim](i,j) << " ";
            }
            std::cout << "]" << std::endl;
        }
        cout << endl;
    }

    void printBoundaryComplex(size_t aDim)
    {
        cout << endl;
        cout << "A unoriented boundary of the complex on dimension " << aDim-1 << ": " << endl;
        MatrixXd myMat = theBoundariesUnoriented[aDim];
        VectorXd myVec(myMat.cols());
        myVec.setOnes();
        MatrixXd myMatMulti = myMat * myVec;
        MatrixXi myFinalMat = myMatMulti.unaryExpr([](int x) { return x % 2; });
        for (long i = 0; i < myMat.rows(); i++)
        {
            if (theSimplicesPerDimOrdered[aDim-1].size() > 0)
            {
                printFunction(unordered_set<Simplex>{theSimplicesPerDimOrdered[aDim-1][i]}, false); 
            }
            std::cout << "  [";
            std::cout << setw(4) << (int)myFinalMat(i) << " ";
            std::cout << "]";
            cout << endl;
        }
        cout << endl;
    }

    void printBoundaryImage(size_t aDim)
    {
        cout << endl;
        cout << "An image (p-chain is span of this) " << aDim-1 << ": " << endl;
        if (theBoundaries[aDim].size() == 0)
        {  
            return; 
        }
        MatrixXd myMat = image(theBoundaries[aDim]);
        for (long i = 0; i < myMat.rows(); i++)
        {
            if (theSimplicesPerDimOrdered[aDim-1].size() > 0)
            {
                printFunction(unordered_set<Simplex>{theSimplicesPerDimOrdered[aDim-1][i]}, false); 
            }
            for (long j = 0; j < myMat.cols(); j++)
            {
                std::cout << "  [";
                std::cout << setw(4) << (int)myMat(i,j) << " ";
                std::cout << "]";
            }
            cout << endl;
        }
        cout << endl;
    }

    void printBoundaryKernel(size_t aDim)
    {
        cout << endl;
        cout << "A kernel " << aDim-1 << ": " << endl;
        if (theBoundaries[aDim].size() == 0 || theSimplicesPerDimOrdered[aDim].size() == 0)
        {  
            return; 
        }
        MatrixXd myMat = kernel(theBoundaries[aDim]);
        for (long i = 0; i < myMat.rows(); i++)
        {
            printFunction(unordered_set<Simplex>{theSimplicesPerDimOrdered[aDim][i]}, false); 
            for (long j = 0; j < myMat.cols(); j++)
            {
                std::cout << "  [";
                std::cout << setw(4) << (int)myMat(i,j) << " ";
                std::cout << "]";
            }
            cout << endl;
        }
        cout << endl;
    }

    size_t printHomology(size_t aDim, bool aPrintMatrix = true)
    {
        cout << endl;
        cout << "A homology " << aDim-1 << ": " << endl;
        if ((theBoundaries[aDim].size() == 0) || (theBoundaries[aDim+1].size() == 0))
        {  
            return 0; 
        }
        MatrixXd myBoundaryDimPlus1 = image(theBoundaries[aDim+1]);
        MatrixXd myKernel = kernel(theBoundaries[aDim]);
        auto myHomologyRows = getRank(myKernel) - getRank(myBoundaryDimPlus1);
        MatrixXd myHomology(myKernel.rows(), myKernel.cols());
        myHomology.setZero(myKernel.rows(), myKernel.cols());
        if (getRank(myBoundaryDimPlus1) > 0)
        {
            // FullPivLU<MatrixXd> mySolver(myBoundaryDimPlus1);
            // MatrixXd myLinearDependence = mySolver.solve(myKernel);
            // MatrixXd myLinearDependenceKernel = kernel(myLinearDependence);
            // myHomology = myKernel * myLinearDependenceKernel;
            // std::vector<VectorXd> homology_basis;
            long j = 0;
            MatrixXd myCurrentBoundary(myKernel.rows(), 1);
            long myCurrentRank = getRank(myBoundaryDimPlus1);
            for (int i = 0; i < myKernel.cols(); ++i) 
            {
                myCurrentBoundary.col(myCurrentBoundary.cols() - 1) = myKernel.col(i);
                MatrixXd augmented(myCurrentBoundary.rows(), myBoundaryDimPlus1.cols() + myCurrentBoundary.cols());
                augmented << myBoundaryDimPlus1, myCurrentBoundary;
                if (getRank(augmented) > myCurrentRank)
                {
                    myCurrentRank = getRank(augmented);
                    MatrixXd myNewBoundary(myCurrentBoundary.rows(), myCurrentBoundary.cols() + 1);
                    myNewBoundary.setZero();
                    for (long f = 0; f < myCurrentBoundary.cols(); f++)
                    {
                        myNewBoundary.col(f) = myCurrentBoundary.col(f);
                    }
                    myCurrentBoundary = myNewBoundary;
                    myHomology.col(j) = myKernel.col(i);
                    j++;
                }
            }
        }
        else
        {
            myHomology = myKernel;
        }
        
        cout << "Rank of kernel: " << getRank(myKernel) << endl;
        cout << "Rank of boundary of dim + 1: " << getRank(myBoundaryDimPlus1) << endl;
        cout << "Expected Homology Rank " << myHomologyRows << endl;
        cout << "Actual Homology Rank " << getRank(myHomology) << endl;
        if (getRank(myHomology) != myHomologyRows)
        {
            throw std::runtime_error("Mismatch In Homology!");
        }
        if (aDim == 0 && ((size_t)myHomologyRows != theConnectedComponents.size()))
        {
            throw std::runtime_error("Mismatch In connected components!");
        }
        theHomologies[aDim-1] = myHomology;
        if (aPrintMatrix)
        {
            for (long i = 0; i < myHomology.rows(); i++)
            {
                if (theSimplicesPerDimOrdered[aDim].size() > 0)
                {
                    printFunction(unordered_set<Simplex>{theSimplicesPerDimOrdered[aDim][i]}, false); 
                }
                for (long j = 0; j < myHomology.cols(); j++)
                {
                    if (myHomology.col(j).squaredNorm() > 0)
                    {
                        std::cout << "  [";
                        std::cout << setw(4) << (int)myHomology(i,j) << " ";
                        std::cout << "]";
                    }
                }
                cout << endl;
            }
        }
        cout << endl;
        return getRank(myHomology);
    }

    void printEulerCharacteristic() 
    {
        std::cout << "Euler Characteristic is: " << getRank(theHomologies[0]) - getRank(theHomologies[1]) + getRank(theHomologies[2]) << std::endl;
    }
};