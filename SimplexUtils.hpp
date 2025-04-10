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

Eigen::MatrixXd readCsv(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;

    std::vector<std::vector<double>> values;
    std::size_t cols = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        bool badRow = false;

        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (std::invalid_argument&) {
                badRow = true;
                break;
            }
        }

        if (!badRow) {
            if (cols == 0) cols = row.size();
            if (row.size() == cols) {
                values.push_back(row);
            }
        }
    }

    Eigen::MatrixXd mat(values.size(), cols);
    for (std::size_t i = 0; i < values.size(); ++i)
        mat.row(i) = Eigen::VectorXd::Map(&values[i][0], cols);

    return mat;
};

double getEuclideanDistance(VectorXd aVec1, VectorXd aVec2)
{
    return (aVec1 - aVec2).norm();
};

void sortRowsByColumn(MatrixXd& mat, int colIndex) {
    std::vector<int> rowIndices(mat.rows());
    std::iota(rowIndices.begin(), rowIndices.end(), 0);

    std::sort(rowIndices.begin(), rowIndices.end(),
        [&mat, colIndex](int a, int b) {
            return mat(a, colIndex) < mat(b, colIndex);  // ascending
        });

    MatrixXd sorted(mat.rows(), mat.cols());
    for (int i = 0; i < mat.rows(); ++i) {
        sorted.row(i) = mat.row(rowIndices[i]);
    }

    mat = sorted;
};

bool isLinearlyIndependent(const MatrixXd& A, const VectorXd& v) {
    MatrixXd Augmented(A.rows(), A.cols() + 1);
    Augmented << A, v;

    FullPivLU<MatrixXd> luA(A);
    FullPivLU<MatrixXd> luAug(Augmented);

    return luAug.rank() > luA.rank();
};
