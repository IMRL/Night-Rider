#include <ros/ros.h>
#include <stdlib.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string>

void pic_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    ros::Time time = img_msg->header.stamp;
    std::string str;
    str = std::to_string(time.sec) + '.' + std::to_string(time.nsec) + ".jpg";
    cv_bridge::CvImageConstPtr ptr;
    ptr = cv_bridge::toCvCopy(img_msg, "bgr8");
    printf("%s\n", str.c_str());
    cv::imwrite(str, ptr->image);
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "img_save");
    ros::NodeHandle n;
    ros::Subscriber sub_image = n.subscribe("/detect_img", 1, pic_callback);   
    ros::spin();
    return 0;
}
