#ifndef UTILS_HPP
#define UTILS_HPP

#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <algorithm>


// Clamp a to be within lim1 and lim2
double clamp(double a, double lim1, double lim2)
{
    double a_min = std::min(lim1, lim2);
    double a_max = std::max(lim1,lim2);
    return std::max(std::min(a,a_max),a_min);
}
// Scales input to be between (0 to 1) based on input limits
double scaleFactor(double f, double tl, double tu)
{
    return (clamp(f,tl,tu)-tl)/(tu-tl);
}

// 旋转矩阵转化为欧拉角
Eigen::Vector3d Rotation2Euler(const Eigen::Matrix3d& R)  //MARKDOWN
{
    Eigen::Vector3d q; 
    double qx,qy,qz;
    if (R(2,0)<1)
    {
        if (R(2,0) > -1)
        {
            qx = std::atan2(R(2,1),R(2,2));
			qy = std::asin(-R(2,0));
			qz = std::atan2(R(1,0),R(0,0));
        }
        else
        {
            qx = 0;
			qy = M_PI/2.0;
			qz = -std::atan2(-R(1,2),R(1,1));
        }
    }
    else
    {
        qx = 0;
		qy = -M_PI/2.0;
		qz = atan2(-R(1,2),R(1,1)); 
    }
    q<<qx,qy,qz;
    return q;

}
//欧拉角转旋转矩阵
Eigen::Matrix3d Euler2Rotation(const Eigen::Vector3d& euler)
{
    Eigen::AngleAxisd Rz(euler(0),Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd Rx(euler(1),Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(euler(2),Eigen::Vector3d::UnitY());
    Eigen::Quaternion<double> q = Rz*Rx*Ry;
    return q.toRotationMatrix();
}

// 基于ZYX 从角速度转换为Euler速率
Eigen::Vector3d AngularVelocity2EulerRates(const Eigen::Vector3d& euler, const Eigen::Vector3d& w)
{
    Eigen::Vector3d eulerRates;
    double t2 = cos(euler(2));
    double t3 = sin(euler(2));
    double t4 = cos(euler(1));
    double t5 = 1.0/t4;
    double t6 = sin(euler(1));
    eulerRates << t5*(t3*w(1)+t2*w(2)), t2*w(1)-t3*w(2), t5*(t4*w(0)+t3*t6*w(1)+t2*t6*w(2));
    return eulerRates;
}
// 基于ZYX将欧拉速率转换为角速度
Eigen::Vector3d EulerRates2AngularVelocity(const Eigen::Vector3d& euler, const Eigen::Vector3d& euler_rates)
{
    Eigen::Vector3d angularVelocity;
	double dqx = euler_rates(2);
	double dqy = euler_rates(1);
	double dqz = euler_rates(0);
	double qx = euler(2);
	double qy = euler(1);
	double t2 = sin(qx);
	double t3 = cos(qx);
	double t4 = cos(qy);
	angularVelocity << dqx-dqz*sin(qy), dqy*t3+dqz*t2*t4, -dqy*t2+dqz*t3*t4;
	return angularVelocity;
}
//计算Vector3D的反对称矩阵
// Eigen::Matrix3d skew(const Eigen::Vector3d& v);
//此处在LieGroup.cpp中有相同函数
// Eigen::Matrix3d skew(const Eigen::Vector3d& v) 
// {
//     Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
//     M<< 0, -v[2] , v[1],
//         v[2] , 0 , -v[0],
//         -v[1] , v[0], 0;
//     return M;
// }

#endif // UTILS_H