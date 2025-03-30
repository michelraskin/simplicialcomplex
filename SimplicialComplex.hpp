#pragma once
#include <iostream>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <string>
#include <Eigen/Dense>
#include "Simplex.hpp"
#include "SimplexUtils.hpp"

using namespace std;
using namespace Eigen;

class SimplicialComplex {
private:
    static constexpr size_t MaxDimension = 3 + 2;
    unordered_set<Simplex> theSimplices;

    std::array<unordered_set<Simplex>, MaxDimension> theSimplicesPerDim;
    std::array<vector<Simplex>, MaxDimension> theSimplicesPerDimOrdered;
    std::array<MatrixXd, MaxDimension> theBoundaries{};
    std::array<MatrixXd, MaxDimension> theBoundariesUnoriented{};

    vector<unordered_set<string>> theConnectedComponents;

public:
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
        for (size_t i = 0; i < 5; i++)
        {
            for (auto myValue : theSimplicesPerDim[i])
            {
                theSimplicesPerDimOrdered[i].push_back(myValue);
            }
            std::stable_sort(theSimplicesPerDimOrdered[i].begin(), theSimplicesPerDimOrdered[i].end());
        }
        for (size_t i = 0; i < 5; i++)
        {
            computeBoundaryMatrix(i, true);
            computeBoundaryMatrix(i, false);
        }
    }

    void computeBoundaryMatrix(size_t aDim, bool aOriented = false)
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
        if (theSimplicesPerDimOrdered[aDim-1].size() == 0)
        {
            size_t i = 0;
            for (auto myValue2 : theSimplicesPerDimOrdered[aDim])
            {
                myBoundary(0,i) = 0;
                i++;
            }
        }
        size_t i = 0;
        for (auto myValue : theSimplicesPerDimOrdered[aDim-1])
        {
            size_t j = 0;
            for (auto myValue2 : theSimplicesPerDimOrdered[aDim])
            {
                const auto& mySubSimplices = myValue2.getSubSimplices();
                auto myIndex = find(mySubSimplices.begin(), mySubSimplices.end(), myValue);
                if (myIndex != mySubSimplices.end())
                {
                    if (aOriented)
                    {
                        size_t index = std::distance(mySubSimplices.begin(), myIndex);
                        myBoundary(i,j) = (myValue2.getSubSimplices().at(index).getOrientation());
                    }
                    else
                    {
                        myBoundary(i,j) = (1);
                    }
                }
                else
                {
                    myBoundary(i,j) = (0);
                }
                j++;
            }
            i++;
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
        printFunction(theSimplicesPerDim[1]);
        cout << endl;

        cout << "Edges: " << endl;
        printFunction(theSimplicesPerDim[2]);
        cout << endl;

        cout << "Triangles: " << endl;
        printFunction(theSimplicesPerDim[3]);
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

    void printHomology(size_t aDim)
    {
        cout << endl;
        cout << "A homology " << aDim-1 << ": " << endl;
        MatrixXd myBoundaryDimPlus1 = image(theBoundaries[aDim+1]);
        MatrixXd myKernel = kernel(theBoundaries[aDim]);
        auto myHomologyRows = getRank(myKernel) - getRank(myBoundaryDimPlus1);
        MatrixXd myHomology(myKernel.rows(), myKernel.cols());
        myHomology.setZero();
        if (getRank(myBoundaryDimPlus1) > 0)
        {
            FullPivLU<MatrixXd> mySolver(myBoundaryDimPlus1);
            MatrixXd myLinearDependence = mySolver.solve(myKernel);
            MatrixXd myLinearDependenceKernel = kernel(myLinearDependence);
            myHomology = myKernel * myLinearDependenceKernel;
        }
        else
        {
            myHomology = myKernel;
        }
        
        cout << "Rank of kernel: " << getRank(myKernel) << endl;
        cout << "Rank of boundary of dim + 1: " << getRank(myBoundaryDimPlus1) << endl;
        cout << "Expected Homology Rank " << myHomologyRows << endl;
        cout << "Actual Homology Rank " << getRank(myHomology) << endl;
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
        cout << endl;
    }
};