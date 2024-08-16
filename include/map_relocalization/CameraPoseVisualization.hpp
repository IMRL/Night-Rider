#ifndef CAMERA_POSE_VISUALIZATION
#define CAMERA_POSE_VISUALIZATION

#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class CameraPoseVisualization {
public:
	CameraPoseVisualization(float r, float g, float b, float a): m_scale(1), m_line_width(0.2){
        m_image_boundary_color.r = r;
        m_image_boundary_color.g = g;
        m_image_boundary_color.b = b;
        m_image_boundary_color.a = a;
        m_optical_center_connector_color.r = r;
        m_optical_center_connector_color.g = g;
        m_optical_center_connector_color.b = b;
        m_optical_center_connector_color.a = a;
    }

    void Eigen2Point(const Eigen::Vector3d& v, geometry_msgs::Point& p) {
        p.x = v.x();
        p.y = v.y();
        p.z = v.z();
    }
	
	// void setImageBoundaryColor(float r, float g, float b, float a=1.0);
	// void setOpticalCenterConnectorColor(float r, float g, float b, float a=1.0);
	void setScale(double s){
        m_scale = s;
    }

	void setLineWidth(double width){
        m_line_width = width;
    }

	void add_pose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q){
        visualization_msgs::Marker marker;

        marker.ns = "";
        marker.id = m_markers.size() + 1;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = m_line_width;

        marker.pose.position.x = 0.0;
        marker.pose.position.y = 0.0;
        marker.pose.position.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;


        geometry_msgs::Point pt_lt, pt_lb, pt_rt, pt_rb, pt_oc, pt_lt0, pt_lt1, pt_lt2;

        Eigen2Point(q * (m_scale *imlt) + p, pt_lt);
        Eigen2Point(q * (m_scale *imlb) + p, pt_lb);
        Eigen2Point(q * (m_scale *imrt) + p, pt_rt);
        Eigen2Point(q * (m_scale *imrb) + p, pt_rb);
        Eigen2Point(q * (m_scale *lt0 ) + p, pt_lt0);
        Eigen2Point(q * (m_scale *lt1 ) + p, pt_lt1);
        Eigen2Point(q * (m_scale *lt2 ) + p, pt_lt2);
        Eigen2Point(q * (m_scale *oc  ) + p, pt_oc);

        // image boundaries
        marker.points.push_back(pt_lt);
        marker.points.push_back(pt_lb);
        marker.colors.push_back(m_image_boundary_color);
        marker.colors.push_back(m_image_boundary_color);

        marker.points.push_back(pt_lb);
        marker.points.push_back(pt_rb);
        marker.colors.push_back(m_image_boundary_color);
        marker.colors.push_back(m_image_boundary_color);

        marker.points.push_back(pt_rb);
        marker.points.push_back(pt_rt);
        marker.colors.push_back(m_image_boundary_color);
        marker.colors.push_back(m_image_boundary_color);

        marker.points.push_back(pt_rt);
        marker.points.push_back(pt_lt);
        marker.colors.push_back(m_image_boundary_color);
        marker.colors.push_back(m_image_boundary_color);

        // top-left indicator
        marker.points.push_back(pt_lt0);
        marker.points.push_back(pt_lt1);
        marker.colors.push_back(m_image_boundary_color);
        marker.colors.push_back(m_image_boundary_color);

        marker.points.push_back(pt_lt1);
        marker.points.push_back(pt_lt2);
        marker.colors.push_back(m_image_boundary_color);
        marker.colors.push_back(m_image_boundary_color);

        // optical center connector
        marker.points.push_back(pt_lt);
        marker.points.push_back(pt_oc);
        marker.colors.push_back(m_optical_center_connector_color);
        marker.colors.push_back(m_optical_center_connector_color);


        marker.points.push_back(pt_lb);
        marker.points.push_back(pt_oc);
        marker.colors.push_back(m_optical_center_connector_color);
        marker.colors.push_back(m_optical_center_connector_color);

        marker.points.push_back(pt_rt);
        marker.points.push_back(pt_oc);
        marker.colors.push_back(m_optical_center_connector_color);
        marker.colors.push_back(m_optical_center_connector_color);

        marker.points.push_back(pt_rb);
        marker.points.push_back(pt_oc);
        marker.colors.push_back(m_optical_center_connector_color);
        marker.colors.push_back(m_optical_center_connector_color);

        m_markers.push_back(marker);
    }

	void reset(){
        m_markers.clear();
    }

	void publish_by(ros::Publisher& pub, const double& stamp){
        visualization_msgs::MarkerArray markerArray_msg;
        for(auto& marker : m_markers) {
		    marker.header.frame_id = "camera_init";
            marker.header.stamp = ros::Time().fromSec(stamp);
		    markerArray_msg.markers.push_back(marker);
        }
    
        pub.publish(markerArray_msg);
    }


private:
	std::vector<visualization_msgs::Marker> m_markers;
	std_msgs::ColorRGBA m_image_boundary_color;
	std_msgs::ColorRGBA m_optical_center_connector_color;
	double m_scale;
	double m_line_width;

	static const Eigen::Vector3d imlt;
	static const Eigen::Vector3d imlb;
	static const Eigen::Vector3d imrt;
	static const Eigen::Vector3d imrb;
	static const Eigen::Vector3d oc  ;
	static const Eigen::Vector3d lt0 ;
	static const Eigen::Vector3d lt1 ;
	static const Eigen::Vector3d lt2 ;
};

const Eigen::Vector3d CameraPoseVisualization::imlt = Eigen::Vector3d(-1.0, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::imrt = Eigen::Vector3d( 1.0, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::imlb = Eigen::Vector3d(-1.0,  0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::imrb = Eigen::Vector3d( 1.0,  0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::lt0 = Eigen::Vector3d(-0.7, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualization::lt1 = Eigen::Vector3d(-0.7, -0.2, 1.0);
const Eigen::Vector3d CameraPoseVisualization::lt2 = Eigen::Vector3d(-1.0, -0.2, 1.0);
const Eigen::Vector3d CameraPoseVisualization::oc = Eigen::Vector3d(0.0, 0.0, 0.0);


#endif