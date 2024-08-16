#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <deque>
#include <vector>

using namespace cv;
using namespace std;

deque<cv::Mat> img_buffer;
deque<double> time_buffer;

void img_cbk(const sensor_msgs::CompressedImage::ConstPtr& msg){

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("fail to load img_msg");
        return;
    }
    cv::Mat img = cv_ptr->image;

    img_buffer.push_back(img);
    time_buffer.push_back(msg->header.stamp.toSec());
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "map_relocalization");
    ros::NodeHandle nh;

    ros::Subscriber sub_img = nh.subscribe("/camera/color/image_raw/compressed", 200000, img_cbk);
    ros::Publisher pub_flow = nh.advertise<sensor_msgs::Image>("/flow_img", 100000);
    ros::Publisher pub_feature = nh.advertise<sensor_msgs::Image>("/feature_img", 100000);

    bool state = ros::ok();
    ros::Rate rate(5000);
    while(state){
        ros::spinOnce();

        Mat frame1, frame2; //定义两个帧
        Mat gray1, gray2; //定义两个灰度图像
        std::vector<Point2f> points1, points2; //定义两个关键点向量

        //读取两个图像
        if(img_buffer.size() >=2){
            frame1 = img_buffer[0];
            frame2 = img_buffer[1];
        }
        else
            continue;

        //将图像转换为灰度图像
        cvtColor(frame1, gray1, COLOR_BGR2GRAY);
        cvtColor(frame2, gray2, COLOR_BGR2GRAY);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(gray1, gray1);
        clahe->apply(gray2, gray2);

        //检测第一个帧的关键点
        goodFeaturesToTrack(gray1, points1, 100, 0.01, 30);

        //计算第一个帧和第二个帧之间的光流
        std::vector<uchar> status;
        std::vector<float> err;
        calcOpticalFlowPyrLK(gray1, gray2, points1, points2, status, err);

        //绘制关键点和运动向量
        for (int i = 0; i < points1.size(); i++)
        {
            if (status[i])
            {
                line(frame1, points1[i], points2[i], Scalar(0, 0, 255));
                circle(frame2, points2[i], 2, Scalar(0, 255, 0), -1);
            }
        }

        cout << "111" << endl;

        //显示结果
        std_msgs::Header head;
        head.stamp = ros::Time().fromSec(time_buffer.front());
        head.frame_id = "camera_init";
        sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(head, "bgr8", frame1).toImageMsg();
        pub_flow.publish(image_msg);

        sensor_msgs::ImagePtr image_msg2 = cv_bridge::CvImage(head, "bgr8", frame2).toImageMsg();
        pub_feature.publish(image_msg2);

        // imshow("frame1", frame1);
        // imshow("frame2", frame2);
        // waitKey(0);

        img_buffer.pop_front();
        time_buffer.pop_front();
    }
    state = ros::ok();
    rate.sleep();

    return 0;
}