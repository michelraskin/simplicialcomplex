#pragma once
#include <iostream>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <string>
#include "Simplex.hpp"

using namespace std;

class SimplicialComplex {
private:
    static constexpr size_t MaxDimension = 3 + 2;
    unordered_set<Simplex> theSimplices;

    std::array<unordered_set<Simplex>, MaxDimension> theSimplicesPerDim;
    std::array<vector<Simplex>, MaxDimension> theSimplicesPerDimOrdered;
    std::array<std::vector<std::vector<int8_t>>, MaxDimension> theBoundaries{};

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
        for (size_t i = 1; i < 4; i++)
        {
            for (auto myValue : theSimplicesPerDim[i])
            {
                theSimplicesPerDimOrdered[i].push_back(myValue);
            }
        }
        for (size_t i = 1; i < 4; i++)
        {
            computeBoundary(i);
        }
    }

    void computeBoundary(size_t aDim, bool aOriented = true)
    {
        // Can refactor here by getting subsimplices
        auto myInitialBoundary = std::vector<std::vector<int8_t>>{};
        if (theSimplicesPerDimOrdered[aDim-1].size() == 0)
        {
            auto myRow = std::vector<int8_t>{};
            for (auto myValue2 : theSimplicesPerDimOrdered[aDim])
            {
                myRow.push_back(0);
            }
            myInitialBoundary.push_back(myRow);
        }
        for (auto myValue : theSimplicesPerDimOrdered[aDim-1])
        {
            auto myRow = std::vector<int8_t>{};
            for (auto myValue2 : theSimplicesPerDimOrdered[aDim])
            {
                const auto& mySubSimplices = myValue2.getSubSimplices();
                auto myIndex = find(mySubSimplices.begin(), mySubSimplices.end(), myValue);
                if (myIndex != mySubSimplices.end())
                {
                    if (aOriented)
                    {
                        size_t index = std::distance(mySubSimplices.begin(), myIndex);
                        myRow.push_back(myValue2.getSubSimplices().at(index).getOrientation());
                    }
                    else
                    {
                        myRow.push_back(1);
                    }
                }
                else
                {
                    myRow.push_back(0);
                }
        }
            myInitialBoundary.push_back(myRow);
        }
        theBoundaries[aDim] = myInitialBoundary;
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
        printFunction(theSimplices);
        cout << endl;

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

    void printBoundary(size_t aDim)
    {
        size_t i = 0;
        cout << endl;
        cout << "A input vector shape: " << endl;
        printFunction(theSimplicesPerDimOrdered[aDim], true);
        cout << endl;
        for (auto myRow : theBoundaries[aDim])
        {
            if (theSimplicesPerDimOrdered[aDim-1].size() > 0)
            {
                printFunction(unordered_set<Simplex>{theSimplicesPerDimOrdered[aDim-1][i]}, false); 
            }
            std::cout << "  [";
            for (auto myElem : myRow)
            {
                std::cout << setw(4) << (int)myElem << " ";
            }
            std::cout << "]" << std::endl;
            i++;
        }
        cout << endl;
    }
};