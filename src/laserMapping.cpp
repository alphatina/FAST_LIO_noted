// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0;  /// kdtree建立时间
double kdtree_search_time = 0.0;  /// kdtree搜索时间
double kdtree_delete_time = 0.0;  /// kdtree删除时间

double T1[MAXN];   /// lidar初始时间戳
double s_plot[MAXN];  /// 整个流程耗时
double s_plot2[MAXN]; /// 特征点数量
double s_plot3[MAXN]; /// kdtree增量时间
double s_plot4[MAXN]; /// kdtree搜索耗时
double s_plot5[MAXN]; /// kdtree删除点数量
double s_plot6[MAXN]; /// kdtree删除耗时
double s_plot7[MAXN]; /// kdtree初始大小
double s_plot8[MAXN]; /// kdtree结束大小
double s_plot9[MAXN]; /// 平均消耗时间
double s_plot10[MAXN]; /// 添加点数量
double s_plot11[MAXN]; /// 点云预处理的总时间

// 统计
double match_time = 0; /// 匹配时间
double solve_time = 0; /// 求解时间
double solve_const_H_time = 0; /// 求解H矩阵时间
int kdtree_size_st = 0; /// ikd-tree获得的节点数
int kdtree_size_end = 0; /// ikd-tree结束时的节点数
int add_point_size = 0;  /// 添加点的数量
int kdtree_delete_counter = 0;  /// 删除点的数量

bool runtime_pos_log = false;  /// 运行时的log是否开启
bool pcd_save_en = false;   /// 是否保存pcd文件
bool time_sync_en = false;  /// 是否同步时间
bool extrinsic_est_en = true;  /// 是否估计外参
bool path_en = true;  /// 是否发布路径的topic
// time_sync_en = true When the time systems of IMU and lidar are different.
// extrinsic_est_en is the switch for extrinsic parameter estimation
/**************************/

float res_last[100000] = {0.0};  /////残差，点到面距离平方和
float DET_RANGE = 300.0f;   ///lidar最大探测距离
const float MOV_THRESHOLD = 1.5f;  ///松弛系数
double time_diff_lidar_to_imu = 0.0;  /// lidar和IMU时间差

mutex mtx_buffer; //mutex 互斥量，不能递归使用，也不能拷贝构造
condition_variable sig_buffer;  // 条件变量

string root_dir = ROOT_DIR;  //设置根目录
string map_file_path;  /// 地图文件路径
string lid_topic;   /// lidar topic
string imu_topic;   /// IMU topic

double res_mean_last = 0.05; // 观测模型中的平均残差
double total_residual = 0.0; // 观测模型中的残差和
double last_timestamp_lidar = 0; // 上一帧最后的lidar接收回调时间戳
double last_timestamp_imu = -1.0;  // 上一帧最后的imu接收回调时间戳
//设置imu的角速度协方差，加速度协方差，角速度偏置的协方差，加速度偏置的协方差
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
/// 角和平面滤波器的大小，地图体素的大小
double filter_size_corner_min = 0.5, filter_size_surf_min = 0.5, filter_size_map_min = 0.5;
double fov_deg = 180;  /// 视场角
double cube_len = 200; /// 局部地图长方体的边长
double HALF_FOV_COS = 0; /// 视场角一半的cos值
double FOV_DEG = 0;   /// 未用
double total_distance = 0; /// 总距离
double lidar_end_time = 0;  /// lidar帧结束时间
double first_lidar_time = 0.0; // lidar帧开始时间戳
int effct_feat_num = 0; /// 有效特征点数
int time_log_counter = 0;  /// log计数器
int scan_count = 0;  /// lidar扫描数
int publish_count = 0;  /// 接收到的IMU的msg总数
int iterCount = 0; /// 迭代计数
int feats_down_size = 0; /// 下采样后的lidar点数
int NUM_MAX_ITERATIONS = 4; /// 最大迭代次数
int laserCloudValidNum = 0; /// 未使用
int pcd_save_interval = -1; /// 每一个PCD文件保存多少个lidar帧（-1表示所有lidar帧都保存在一个PCD文件中）
int pcd_index = 0;   /// pcd计数
bool point_selected_surf[100000] = {0}; //是否是平面特征点
bool lidar_pushed;  /// 用于判断是否将lidar数据放入meas中 
bool flg_first_scan = true;  /// 是否是第一帧
bool flg_exit = false;  /// 退出程序
bool flg_EKF_inited;  /// EKF是否已经初始化
bool scan_pub_en = false;  /// 是否发布当前正在扫描的点云的topic
bool dense_pub_en = false; ///// 是否发布经过运动畸变校正并注册到IMU坐标系的点云的topic，
bool scan_body_pub_en = false; /// 是否发布经过运动畸变校正并注册到IMU坐标系的点云的topic，需要dense_pub_en同时为true才发布

vector<vector<int>>  pointSearchInd_surf;  //每个点的索引,暂时没用到
vector<BoxPointType> cub_needrm;  ///// ikd-tree中需要移除的局部地图边界
vector<PointVector>  Nearest_Points;  //每个点的最近点序列
vector<double>       extrinT(3, 0.0);  /// 平移外参
vector<double>       extrinR(9, 0.0);  /// 旋转外参
deque<double>                     time_buffer; // lidar数据时间戳缓存队列
deque<PointCloudXYZI::Ptr>        lidar_buffer; //// 记录特征提取或降采样后的lidar（特征）数据
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;  // IMU数据队列

// XYZI: X/Y/Z/Intensity
// Ptr: pointer type
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI()); //提取地图中的特征点，IKD-tree获得
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());  //去畸变后的lidar特征
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI()); //去畸变后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); //去畸变后降采样的单帧点云，世界系
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));  //特征点在地图中对应点的局部平面法向量,世界系
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); //去畸变后降采样的单帧点云，body系
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); //特征点在地图中对应点的局部平面法向量
PointCloudXYZI::Ptr _featsArray;  // ikd-tree中，map需要移除的点云序列

// VoxelGrid:使用体素化网格的方法实现下采样，并保持点云的形状特征
pcl::VoxelGrid<PointType> downSizeFilterSurf;   /////单帧内降采样使用voxel grid
pcl::VoxelGrid<PointType> downSizeFilterMap;   //未使用

KD_TREE<PointType> ikdtree;  // ikd-tree类

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);  /////未使用
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);  
V3D euler_cur;   //当前的欧拉角
V3D position_last(Zero3d);   //上一帧的位置
V3D Lidar_T_wrt_IMU(Zero3d); //Translation matrix between lidar and IMU
M3D Lidar_R_wrt_IMU(Eye3d); //Rotation matrix between lidar and IMU

/*** EKF inputs and output ***/
MeasureGroup Measures; // lidar and IMU data
/*** 
 * state_ikfom: 22 dimension
 * noise：12 dimension
 * input_ikfom:6 dimension
***/
esekfom::esekf<state_ikfom, 12, input_ikfom> kf; /// 误差状态，噪声维度是12，IMU输入
state_ikfom state_point; // 24维状态
vect3 pos_lid; // 世界系下的lidar坐标

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped; //Odometry message
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());  // 定义指向激光雷达数据的预处理类Preprocess的智能指针
shared_ptr<ImuProcess> p_imu(new ImuProcess()); // 定义指向IMU数据预处理类ImuProcess的智能指针

/// 中断处理，退出程序
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    /// 唤醒所有等待队列中阻塞的线程 线程被唤醒后，会通过轮询方式获得锁，
    /// 只有一个线程可以获得锁，其他线程等待锁，不会被再次阻塞。
    sig_buffer.notify_all();  
}

/////将fast lio2信息打印到log中
inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

// 把lidar点从body系转换到世界系
void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// 把lidar点从body系转换到世界系
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

/// 把lidar点从body系转换到世界系
template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

/// 含有RGB的点云从body系转到world系
void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// 点云从Lidar系转到IMU系
void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

// 得到被剔除的点
void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history); //返回被剔除的点
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

// 在拿到eskf前馈结果后，动态调整局部地图的区域
BoxPointType LocalMap_Points;  // ikd-tree中局部地图的三维边界点
bool Localmap_Initialized = false; // 局部地图是否初始化
// Segment the map in lidar FOV & update local map
void lasermap_fov_segment()
{
    cub_needrm.clear();  // 清空需要移除的区域
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    

    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);

    /// 世界系下的lidar位置
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        // 初始化局部地图的大小和位置
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0; //cube_len = 200; vertex 顶点
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // 各个方向上Lidar与局部地图边界的距离，或者说是lidar与局部地图立方体盒子六个面的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离小于阈值，标记需要移动局部地图
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    /// 如果不需要移动局部地图，则直接退出本函数
    if (!need_move)
        return;

    /// 新的局部地图边界点
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    /// 移动距离是 (MOV_THRESHOLD-1)*DET_RANGE
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    /// 删除cub_needrm范围内的点
    if(cub_needrm.size() > 0) 
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

// 除AVIA类型之外的lidar点云回调函数
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock(); // 加锁
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);  /// 点云预处理
    lidar_buffer.push_back(ptr);  /// 点云放入缓冲区
    time_buffer.push_back(msg->header.stamp.toSec());  /// 时间戳放入缓冲区
    last_timestamp_lidar = msg->header.stamp.toSec();  /// //记录时间戳
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// time difference between the lidar and IMU
double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
// livox lidar点云回调函数
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime(); 
    scan_count ++;  //扫描数+1
    // 如果当前lidar扫描时间戳比上一次lidar扫描时间戳早，需要将激光雷达数据缓存队列清空
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    // 如果不需要进行时间同步，而imu时间戳和雷达时间戳相差大于10s，则输出错误信息
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }
    // time_sync_en为true时，当imu时间戳和雷达时间戳相差大于1s时，进行时间同步
    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr); 
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// callback function for IMU data
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // 将IMU和激光雷达点云的时间戳对齐
    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock(); //上锁，同一时刻只能有一个线程持有该锁，拿不到锁就会阻塞，等待notify_all或者notify_one唤醒
    // 如果当前IMU的时间戳小于上一个时刻IMU的时间戳，则IMU数据有误，将IMU数据缓存队列清空
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock(); //解锁
    sig_buffer.notify_all(); //唤醒等待队列中所有阻塞的线程，但只有一个线程能够获得锁，其余的线程会继续尝试获得锁。
}

double lidar_mean_scantime = 0.0; //// lidar扫描一帧平均时间
int    scan_num = 0; 
// 取一帧lidar数据，以及对应时间区间的IMU数据，保存到meas中
// 备注：必须同时有IMU数据和lidar数据
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front(); //一帧lidar数据的起始时间
        // 如果该lidar点太少，这帧无效
        if (meas.lidar->points.size() <= 1) 
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) ////如果扫描时间太短，这帧无效
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);  /// lidar的时间单位是us，转换到ms
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }
    // 最新的IMU时间戳(也就是队尾的)不能早于雷达的end时间戳，因为last_timestamp_imu比较时是加了0.1的，所以要比较大于雷达的end时间戳
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    // 拿出lidar_beg_time到lidar_end_time之间的所有IMU数据
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;  //将lidar_pushed置为false，等待下一次的数据
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd; // 需要在地图中新增的雷达点
    PointVector PointNoNeedDownsample; // 不需要在地图中新增的雷达点
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            /// NUM_MATCH_POINTS = 5；
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else // 初始点
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));  /// 存储等待发布的点云
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());   /// 存储等待保存的点云
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)   /// 是否发布扫描点云
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);   /// 是否需要降采样
        int size = laserCloudFullRes->points.size();   /// 点云大小
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));  /// 存储转换到世界坐标系的点云

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}
//把去畸变的雷达系下的点云转到IMU系
void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    //转换到IMU坐标系
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}
//把起作用的特征点转到地图中
void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}
// 发布ikd-tree地图
void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}
// 设置输出的t,q，在publish_odometry，publish_path调用
template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}
//发布里程计
void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        //设置协方差 P里面先是旋转后是位置 这个POSE里面先是位置后是旋转
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    //发布tf变换
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}
//每隔10个发布一下位姿
void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

// 观测模型
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif

    // travel throughout all feature points
    for (int i = 0; i < feats_down_size; i++)
    {
        // feats_down_body: lidar坐标系下降采样后的lidar特征点 
        PointType &point_body  = feats_down_body->points[i]; 
        // feats_down_world: world坐标系下降采样后的lidar特征点 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        //// lidar coordinate -> IMU coordinate -> world coordinate 
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        /// NUM_MATCH_POINTS: 5
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            //// Similar to the LOAM series, determine whether it is a valid match point
            //// 要求特征点在地图中的近邻点数量大于5，而且到这些近邻点距离小于阈值，否则该点不是有效特征点
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;  //法向量
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f)) // Calculate the normal vector of closet surface
        {
            // residual is the distance from the point to the plane
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            /// points with small residuals are effective points
            /// Residuals exceeding this threshold are either outliers or newly observed points. Discard points with too large residuals.
            /// 发射距离越长，测量误差越大，用lidar坐标系下的发射距离p_body.norm()归一化，消除lidar点发射距离的影响
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2; // residual
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;
    /// 只保留有效特征点
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];   /// 存储有效点坐标
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;   /// 残差平均值
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    /// h_x is the Measuremnt Jacobi matrix of residual h with respect to error state x. See Formula(14) in FAST-LIO.
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); 
    /// h is the residual
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i]; /// 有效点坐标
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);   /// 从列向量到三维反对称矩阵
        /// lidar coordinate to IMU coordinate
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);  /// 计算IMU坐标系下的三维反对称矩阵

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        /// transform to IMU coordinate
        V3D C(s.rot.conjugate() *norm_vec); /// rot的转置与法向量相乘得到C
        V3D A(point_crossmat * C); // imu坐标系下的点坐标的反对称 点乘 C 得到A
        if (extrinsic_est_en) //Optimize Lidar-IMU extrinsic parameters
        {
            //带be的是Lidar原始坐标系的点坐标，不带be的是IMU坐标系的Lidar点坐标
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else //don't optimize Lidar-IMU extrinsic parameters.
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measurement: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true); // 高密度点云
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4); // IEKF最大迭代次数
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5); // 地图分割，每个体素的边长
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180); //视场角
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16); // 激光雷达线束数
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, true);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true); //打开外参估计
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); //IMU和lidar之间的平移矩阵，杆臂
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>()); //IMU和Lidar之间的旋转矩阵，旋转角
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
     /** 变量定义
     * effect_feat_num          （后面的代码中没有用到该变量）
     * frame_num                雷达总帧数
     * deltaT                   （后面的代码中没有用到该变量）
     * deltaR                   （后面的代码中没有用到该变量）
     * aver_time_consu          每帧平均的处理总时间
     * aver_time_icp            每帧中icp的平均时间
     * aver_time_match          每帧中匹配的平均时间
     * aver_time_incre          每帧中ikd-tree增量处理的平均时间
     * aver_time_solve          每帧中计算的平均时间
     * aver_time_const_H_time   每帧中计算的平均时间（当H恒定时）
     * flg_EKF_converged        （后面的代码中没有用到该变量）
     * EKF_stop_flg             （后面的代码中没有用到该变量）
     * FOV_DEG                  （后面的代码中没有用到该变量）
     * HALF_FOV_COS             （后面的代码中没有用到该变量）
     * _featsArray              （后面的代码中没有用到该变量）
     **/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    /// FOV_DEG也没用到
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos(FOV_DEG * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    /// 将数组point_selected_surf内元素的值全部设为true，数组point_selected_surf用于选择平面点
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    /// VoxelGrid滤波器参数，即滤波时创建的体素边长为filter_size_surf_min
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    // VoxelGrid滤波器参数，即滤波时创建的体素边长为filter_size_map_min
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    /// lidar相对于IMU的外参R和T
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);

    /// 设置IMU的参数，p_imu初始化，p_imu是ImuProcess的智能指针（ImuProcess是IMU处理的类）
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    //设置epsi数组全部为0.001
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);

    //接收特定系统的模型及其差异，作为一个维数变化的特征矩阵进行测量。
    //通过函数h_dyn_share_in同时计算测量（z）、估计测量（h）、偏微分矩阵（h_x，h_v）和噪声协方差（R）。
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    // lidar点云的订阅器sub_pcl，订阅点云的topic
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    // 发布当前正在扫描的点云，topic名字为/cloud_registered
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    // 发布经过运动畸变校正注册到IMU坐标系的点云，topic名字为/cloud_registered_body
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    // 后面的代码中没有用到
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    // 后面的代码中没有用到
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    // 发布当前里程计信息，topic名字为/Odometry
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    // 发布里程计总的路径，topic名字为/path
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
//------------------------------------------------------------------------------------------------------
   
    // 中断处理函数，如果有中断信号（比如Ctrl+C），则执行第二个参数里面的SigHandle函数
    signal(SIGINT, SigHandle);
    /// 设置监听频率：5000 Hz，则ROS程序主循环每次运行的时间至少为0.2 ms
    ros::Rate rate(5000);
    bool status = ros::ok();

    // 程序主循环
    while (status)
    {
        // 如果有中断产生，则结束主循环
        if (flg_exit) break;
        // ROS消息回调处理函数，放在ROS的主循环中
        ros::spinOnce();
        /// 处理IMU和lidar接收buffer的数据，对齐时间，并放进Measures中
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)  //第一帧lidar数据
            {
                // 记录激光雷达第一次扫描的时间
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            // 对IMU数据进行预处理，其中包含了初始化，点云畸变处理，前向传播，反向传播
            p_imu->Process(Measures, kf, feats_undistort);

            state_point = kf.get_x();
            /// pos_lid: 世界系下lidar位置； state_point.pos: 世界系下IMU位置
            //下面式子的意义是W^p_L = W^p_I + W^R_I * I^p_L
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;  //无点则跳过
            }

            //判断是否初始化完成，需要满足第一次扫描的时间和第一个点云时间的差值大于INIT_TIME
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment(); /// 调整局部地图

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);  //获得去畸变后的点云数据
            downSizeFilterSurf.filter(*feats_down_body);  //滤波降采样后的点云数据
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();  //滤波后的点云数量
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);  
                    //将下采样得到的地图点转换为世界坐标系下的点云
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            // 获取ikd tree中的有效节点数，无效点就是被打了deleted标签的点
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            //// 外参旋转矩阵转欧拉角
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                /// 清空PCL_Storage
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);   ////搜索索引
            Nearest_Points.resize(feats_down_size);    ////降采样后的点云用于搜索最近点
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            /// IEKF核心函数
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            // update KD-tree
            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;   //// 整个流程的总时间
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
