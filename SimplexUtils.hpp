#pragma once
#include <iostream>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <string>
#include <functional>
#include <Eigen/Dense>
#include "Simplex.hpp"
using namespace std;
using namespace Eigen;

template <>
struct hash<Simplex> {
    size_t operator()(const Simplex& aSimplex) const {
        size_t hashValue = 0;
        for (const auto& vertex : aSimplex.getUnorientedSimplex()) {
            hashValue ^= hash<string>{}(vertex) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
        }
        return hashValue;
    }
};

void rowEchelon(MatrixXd &mat) {
    int rows = mat.rows();
    int cols = mat.cols();
    int lead = 0;

    for (int r = 0; r < rows; ++r) {
        if (lead >= cols)
            return;

        int i = r;
        while (mat(i, lead) == 0) {
            i++;
            if (i == rows) {
                i = r;
                lead++;
                if (lead == cols)
                    return;
            }
        }

        // Swap rows i and r
        mat.row(i).swap(mat.row(r));

        // Normalize pivot row
        mat.row(r) /= mat(r, lead);

        // Eliminate all rows below
        for (int j = r + 1; j < rows; ++j) {
            mat.row(j) -= mat(j, lead) * mat.row(r);
        }

        lead++;
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