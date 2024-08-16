#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <deque>
#include <vector>
#include "map_relocalization/BoundingBoxes.h"
#include <sstream>

using namespace std;

deque<cv::Mat> img_buffer;
deque<double> time_buffer;
deque<map_relocalization::BoundingBoxesPtr> box_buffer;
deque<double> box_time_buffer;

vector<cv::Rect> box1, box2;

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

void box_cbk(const map_relocalization::BoundingBoxes::ConstPtr& msg){
    map_relocalization::BoundingBoxes::Ptr new_msg(new map_relocalization::BoundingBoxes(*msg));

    box_buffer.push_back(new_msg);
    box_time_buffer.push_back(msg->header.stamp.toSec());
}

int main(int argc, char** argv){
    ros::init(argc, argv, "map_relocalization");
    ros::NodeHandle nh;

    ros::Subscriber sub_img = nh.subscribe("/camera/color/image_raw/compressed", 200000, img_cbk);
    ros::Subscriber sub_box = nh.subscribe("/yolov7_bbox", 200000, box_cbk);
    ros::Publisher pub_detect_img = nh.advertise<sensor_msgs::Image>("/detect_img", 100000);
    ros::Publisher pub_bin_img = nh.advertise<sensor_msgs::Image>("/binary_image", 100000);
    ros::Publisher pub_grey_img = nh.advertise<sensor_msgs::Image>("/grey_image", 100000);
    ros::Publisher pub_box_img = nh.advertise<sensor_msgs::Image>("/box_img", 100000);

    bool status = ros::ok();
    ros::Rate rate(5000);
    while(status){
        ros::spinOnce();
        if(!img_buffer.empty() && !box_buffer.empty()){
            cv::Mat img = img_buffer.front();
            cv::Mat img_copy = img.clone();
            cv::Mat img_copy_copy = img.clone();

            cv::Mat grey_img(img.rows, img.cols, CV_8UC1);
            cv::cvtColor(img, grey_img, cv::COLOR_BGR2GRAY);
            cv::Mat bin_img;
            cv::threshold(grey_img, bin_img, 250, 255, cv::THRESH_BINARY);

            vector<cv::Rect> tmp_boxes;
            vector<vector<cv::Point>> contours;
            cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            for(int i = 0; i < contours.size(); i++){
                cv::Rect rect = cv::boundingRect(contours[i]);
                // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;

                int area = rect.width * rect.height;

                if(area < 20 || rect.width < 4 || rect.height < 4 || rect.height / rect.width > 6)
                    continue;

                rect.x = rect.x - 4;
                rect.width = rect.width + 8;
                rect.y = rect.y - 4;
                rect.height = rect.height + 8;

                int light_nei_v = 0;
                for(int v = rect.y; v < rect.y + rect.height; ++v){
                    float light_intensity = grey_img.ptr<uchar>(v)[rect.x + rect.width / 2];
                    if(light_intensity > 245) light_nei_v++;
                }
                if(light_nei_v / float(rect.width + 1) < 0.8 && area > 400)
                    continue;

                if(rect.x >= 0 && rect.y >= 0 && rect.width < bin_img.cols && rect.height < bin_img.rows){
                    tmp_boxes.push_back(rect);
                    // cv::rectangle(img, rect, cv::Scalar(255, 255, 255), 3);
                }
            }

            auto cur_boxes = box_buffer.front();
            vector<vector<cv::Point>> boxes;
            for (int i = 0; i < cur_boxes->bounding_boxes.size(); i++){
                vector<cv::Point> box;
                box.push_back(cv::Point(cur_boxes->bounding_boxes[i].xmin,cur_boxes->bounding_boxes[i].ymin));
                box.push_back(cv::Point(cur_boxes->bounding_boxes[i].xmax,cur_boxes->bounding_boxes[i].ymin));
                box.push_back(cv::Point(cur_boxes->bounding_boxes[i].xmax,cur_boxes->bounding_boxes[i].ymax));
                box.push_back(cv::Point(cur_boxes->bounding_boxes[i].xmin,cur_boxes->bounding_boxes[i].ymax));
                boxes.push_back(box);
                
                cv::Rect rect(box[0], box[2]);
                box2.push_back(rect);
            }
            cv::drawContours(img_copy, boxes, -1, cv::Scalar(255, 0, 0), 3);
            cv::drawContours(img_copy_copy, boxes, -1, cv::Scalar(255, 0, 0), 3);

            //去除重复的检测框
            auto iter_box1 = tmp_boxes.begin();
            while(iter_box1 != tmp_boxes.end()){
                bool find_rep = false;
                for(auto iter_box2 = box2.begin(); iter_box2 != box2.end(); ++iter_box2){
                    cv::Rect old_box(*iter_box2);
                    if(((*iter_box1) & old_box).area() > 0.25 * iter_box1->area() || ((*iter_box1) & old_box).area() > 0.25 * old_box.area()){
                        find_rep = true;
                        break;
                    }
                }
                if(find_rep){
                    iter_box1 = tmp_boxes.erase(iter_box1);
                }
                else{
                    ++iter_box1;
                }
            }

            for(int i = 0; i < tmp_boxes.size(); ++i){
                cv::Rect rec = tmp_boxes[i];
                cv::rectangle(img_copy_copy, rec, cv::Scalar(0, 0, 255), 3);
            }


            // for (int i = 0; i < box1.size(); i++){
            //     cv::Rect rect1 = box1[i];
            //     for (int j = 0; j < box2.size(); j++){
            //         cv::Rect rect2 = box2[j];
            //         int bin_min_u = 10000, bin_min_v = 10000, bin_max_u = -10000, bin_max_v = -10000;
            //         if((rect1 & rect2).area()){
            //             cv::Rect union_rect = rect1 | rect2;
            //             for (int u = union_rect.x; u < union_rect.x + union_rect.width; u++){
            //                 for (int v = union_rect.y; v < union_rect.y + union_rect.height; v++){
            //                     float intensity;
            //                     intensity = grey_img.ptr<uchar>(v)[u];
                        
            //                     if (intensity > 252){
            //                         int light_nei = 0;
            //                         for(int du = -3; du < 4; du++)
            //                             for(int dv = -3; dv < 4; dv++){
            //                                 float light_intensity;
            //                                 light_intensity = grey_img.ptr<uchar>(v + dv)[u + du];
            //                                 if (light_intensity > 249) light_nei++;
            //                             }
            //                         if(light_nei < 15) continue;
            //                         bin_min_u = min(bin_min_u, u);
            //                         bin_min_v = min(bin_min_v, v);
            //                         bin_max_u = max(bin_max_u, u);
            //                         bin_max_v = max(bin_max_v, v);
            //                     }
            //                 }
            //             }
            //         }
            //         else
            //             continue;
            //         cv::Rect rect3(cv::Point(bin_min_u, bin_min_v), cv::Point(bin_max_u, bin_max_v));
            //         cv::rectangle(img_copy, rect3, cv::Scalar(0, 255, 0), 6);
            //     }
            // }

            // cv::imshow("rect", img_copy);
            // cv::waitKey(0.033);
            std_msgs::Header head;
            head.stamp = ros::Time().fromSec(time_buffer.front());
            head.frame_id = "camera_init";
            sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(head, "bgr8", img_copy).toImageMsg();
            pub_detect_img.publish(image_msg);

            stringstream ss;
            ss << setprecision(16) << time_buffer.front();
            cv::imwrite(ss.str() + ".png", img_copy);
            cv::imwrite(ss.str() + "_bin.png", img_copy_copy);

            // std_msgs::Header head2;
            // head2.stamp = ros::Time().fromSec(time_buffer.front());
            // head2.frame_id = "camera_init";
            // sensor_msgs::ImagePtr image_msg2 = cv_bridge::CvImage(head2, "mono8", grey_img).toImageMsg();
            // pub_bin_img.publish(image_msg2);

            // std_msgs::Header head3;
            // head3.stamp = ros::Time().fromSec(time_buffer.front());
            // head3.frame_id = "camera_init";
            // sensor_msgs::ImagePtr image_msg3 = cv_bridge::CvImage(head3, "mono8", bin_img).toImageMsg();
            // pub_grey_img.publish(image_msg3);

            time_buffer.pop_front();
            img_buffer.pop_front();
            box_buffer.pop_front();
            box_time_buffer.pop_front();
            // box1.clear();
            box2.clear();
        }
        status = ros::ok();
        rate.sleep();
    }
}