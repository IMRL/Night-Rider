#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include <GeographicLib/LocalCartesian.hpp>

using namespace std;

int main(){
    Eigen::Vector3d init_lla, gps_lla;
    init_lla << 22.126760, 113.546800, 4.79900;
    gps_lla << 22.121136, 113.546555, 5.613000;
    // gps_lla << 22.131609, 113.544002, 1.727000;
    GeographicLib::LocalCartesian local_cartersian;
    local_cartersian.Reset(init_lla(0), init_lla(1), init_lla(2));

    double x, y, z;
    local_cartersian.Forward(gps_lla(0), gps_lla(1), gps_lla(2), x, y, z);
    cout << "x: " << x << " y: " << y << " z: " << z << endl;
    double lio_x, lio_y;
    lio_x = cos(-1.1406) * x + sin(-1.1406) * y;
    lio_y = -sin(-1.1406) * x + cos(-1.1406) * y;
    cout << "lio_x: " << lio_x << " lio_y: " << lio_y << endl;
    return 0;
}

// x: 1.65065 y: -0.332196 z: 1.249
// x: -1.96017 y: -151.149 z: -1.2348