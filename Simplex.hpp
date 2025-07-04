#pragma once
#include <iostream>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <string>
#include <functional>

using namespace std;

class Simplex
{
    unordered_set<string> theUnorientedSimplex;
    vector<string> theOrientedSimplex;
    int64_t theOrientation;
    set<Simplex> theSubSimplices;

public:
    Simplex(vector<string> aOrientedSimplex, int64_t aOrientation = 1)
    {
        theOrientation = aOrientation;
        for (const auto& myVertex : aOrientedSimplex)
        {
            theUnorientedSimplex.insert(myVertex);
            theOrientedSimplex.push_back(myVertex);
        }
        std::stable_sort(theOrientedSimplex.begin(), theOrientedSimplex.end());
        for (size_t i = 0; i < size(); i++)
        {
            auto mySubSimplex = eject(i);
            theSubSimplices.insert(Simplex(mySubSimplex, (i % 2) ? 1 : -1));
        }
    }

    vector<string> eject(size_t aElem)
    {
        auto myVector = std::vector<string>{};
        for (size_t i = 0; i < size(); i++)
        {
            if (i != aElem)
            {
                myVector.push_back(theOrientedSimplex[i]);
            }
        }
        return myVector;
    }

    const auto& getUnorientedSimplex() const
    {
        return theUnorientedSimplex;
    }

    const auto& getOrientedSimplex() const
    {
        return theOrientedSimplex;
    }

    const auto& getSubSimplices() const
    {
        return theSubSimplices;
    }

    int64_t getOrientation() const
    {
        return theOrientation;
    }

    bool operator==(const Simplex& aOtherSimplex) const 
    {
        return theUnorientedSimplex == aOtherSimplex.getUnorientedSimplex();
    }

    bool operator<(const Simplex& aOtherSimplex) const 
    {
        return theOrientedSimplex < aOtherSimplex.getOrientedSimplex();
    }

    size_t size() const
    {
        return theOrientedSimplex.size();
    }

    auto begin() const
    {
        return theOrientedSimplex.begin();
    }

    auto end() const
    {
        return theOrientedSimplex.end();
    }

    bool contains(const Simplex& aOtherSimplex) const
    {
        for (const auto& element : aOtherSimplex.getUnorientedSimplex())
        {
            if (theUnorientedSimplex.find(element) == theUnorientedSimplex.end()) 
            {
                return false; 
            }
        }
        return true; 
    }
};

std::ostream& operator<<(std::ostream& out, const Simplex& aSimplex) {
    out << "{";
    for (const auto& myValue : aSimplex) {
        out << setw(8) << myValue << " ";
    }
    out << "}";
    return out;
}