#pragma once
#include <iostream>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <string>
#include <functional>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "Simplex.hpp"
using namespace std;
using namespace Eigen;

template <>
struct hash<Simplex> {
    size_t operator()(const Simplex& aSimplex) const {
        size_t hashValue = 0;
        for (const auto& vertex : aSimplex.getOrientedSimplex()) {
            hashValue ^= std::hash<std::string>{}(vertex) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
        }

        return hashValue;
    }
};

MatrixXd kernel(const MatrixXd& A) {
    FullPivLU<MatrixXd> myLuDecomp(A);
    return myLuDecomp.kernel();  
};

MatrixXd image(const MatrixXd& A) {
    FullPivLU<MatrixXd> myLuDecomp(A);
    return myLuDecomp.image(A); 
};

MatrixXd rowEchelon(const MatrixXd& A) {
    FullPivLU<MatrixXd> myLuDecomp(A);
    MatrixXd U = myLuDecomp.matrixLU().triangularView<Upper>(); 
    return U;
};

int getRank(const MatrixXd& A) {
    return A.fullPivLu().rank();
};

int getNullity(const MatrixXd& A) {
    return A.cols() - A.fullPivLu().rank();
};

Eigen::MatrixXd readCsv(const std::string& aFileName) {
    std::ifstream myFile(aFileName);
    std::string myLine;

    std::vector<std::vector<double>> myValue;
    std::size_t myCols = 0;

    while (std::getline(myFile, myLine)) 
    {
        std::stringstream mySs(myLine);
        std::string myCell;
        std::vector<double> myRow;
        bool myBadRow = false;

        while (std::getline(mySs, myCell, ',')) 
        {
            try 
            {
                myRow.push_back(std::stod(myCell));
            } 
            catch (std::invalid_argument&) 
            {
                myBadRow = true;
                break;
            }
        }

        if (!myBadRow) 
        {
            if (myCols == 0) myCols = myRow.size();
            if (myRow.size() == myCols) 
            {
                myValue.push_back(myRow);
            }
        }
    }

    Eigen::MatrixXd myMat(myValue.size(), myCols);
    for (std::size_t i = 0; i < myValue.size(); ++i)
        myMat.row(i) = Eigen::VectorXd::Map(&myValue[i][0], myCols);

    return myMat;
};

double getEuclideanDistance(VectorXd aVec1, VectorXd aVec2)
{
    return (aVec1 - aVec2).norm();
};

bool isLinearlyIndependent(const MatrixXd& A, const VectorXd& v) 
{
    MatrixXd myAugmented(A.rows(), A.cols() + 1);
    myAugmented << A, v;

    FullPivLU<MatrixXd> myLuDecompA(A);
    FullPivLU<MatrixXd> myLuDecompAug(myAugmented);

    return myLuDecompAug.rank() > myLuDecompA.rank();
};