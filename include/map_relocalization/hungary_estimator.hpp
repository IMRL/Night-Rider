#ifndef HUNGARY_ESTIMATOR_HPP
#define HUNGARY_ESTIMATOR_HPP

#include<iostream>
#include<algorithm>
#include<Eigen/Core>
#include<vector>

using namespace std;

class Hungary{
public:
    template<typename Derived>
    Hungary(int num, const Eigen::MatrixBase<Derived>& costMatrix): n(num), mat(costMatrix), matRcd(costMatrix){
        assign.resize(n);
        for (int i = 0; i < num; i++)
            assign[i] = -1;
        totalCost = 0;        
    }

    void rowSub(); //行规约

    void colSub(); //列规约

    bool isOptimal(); //判断是否通过试指派找到最优分配方案

    void matTrans(); //矩阵变换，使代价矩阵出现足够的0元素

    vector<int> solve();

private:
    int n;//元素个数
    vector<int> assign;//分配结果
    Eigen::MatrixXd mat;//代价矩阵
    Eigen::MatrixXd matRcd;//代价矩阵
    double totalCost;//总成本
};

#endif