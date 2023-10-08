#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>

using namespace std;

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE{AVIA = 1, VELO16, OUST64}; //{1, 2, 3}
enum TIME_UNIT{SEC = 0, MS = 1, US = 2, NS = 3};

/// 特征点类型
enum Feature{
  Nor, /// 未判定，初始状态
  Poss_Plane, /// 可能的平面点
  Real_Plane, /// 确定的平面点
  Edge_Jump, /// 跳变的边缘点
  Edge_Plane, /// 平面的边缘
  Wire, /// 线
  ZeroPoint ///无效点，程序中未使用
  };

enum Surround{Prev, Next};

/// 折线点类型
enum E_jump{
  Nr_nor, /// 正常点
  Nr_zero, /// 入射方向和平面的夹角接近0度
  Nr_180, /// 入射方向和平面的夹角接近180度
  Nr_inf, /// 距离过远
  Nr_blind  /// 靠近盲区
  };

struct orgtype
{
  double range; /// 点到lidar的水平距离（三维空间距离在XY平面的投影）
  double dista; /// 当前点与后一个点之间的距离
  //假设雷达原点为O，前一个点为M，当前点为A，后一个点为N
  double angle[2]; /// 角OAM和角OAN的cos值
  double intersect; /// 角MAN的cos值
  E_jump edj[2];  /// 前后两点的类型
  Feature ftype;  /// 当前点类型
  orgtype()
  {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;  /// XYZ+padding
      float intensity;
      float time;  
      uint16_t ring;   /// 点所属的圈数
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW  ///内存对齐
  };
}  // namespace velodyne_ros

/// velodyne_ros的配对点
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (uint16_t, ring, ring)
)

namespace ouster_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint32_t t;
      uint16_t reflectivity;  /// 反射率
      uint8_t  ring;
      uint16_t ambient;
      uint32_t range;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace ouster_ros

// clang-format off
/// ouster_ros的配对点
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

/// lidar点云数据预处理
class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();
  
  // 处理Livox lidar数据
  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  // 处理OUST64/velodyne lidar数据
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_corn, pl_surf;  /// 全部点/角点/平面点
  PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
  vector<orgtype> typess[128]; //maximum 128 line lidar
  float time_unit_scale;
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;  ///lidar类型，采样间隔，线数，扫描频率，扫描点时间单位
  double blind;  /// 最小距离阈值，小于该阈值是盲区，无效点
  bool feature_enabled, given_offset_time;  ///是否提取特征，是否有时间偏移
  ros::Publisher pub_full, pub_surf, pub_corn;  ///发布全部点/角点/平面点
    

  private:
  // 处理Livox lidar数据 统一用pointCloud  XYZI ring timestap
  void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
  // 处理oust64 lidar数据
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  // 处理velodyne lidar数据
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  void pub_func(PointCloudXYZI &pl, const ros::Time &ct);
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct); /// 未使用
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
  int group_size;
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  double vx, vy, vz;
};
