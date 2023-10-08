#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

/// 判断数据点的时间顺序
const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_; /// 没有去除畸变之前的lidar数据
  sensor_msgs::ImuConstPtr last_imu_; /// 一帧lidar数据对应的最后一个imu数据
  deque<sensor_msgs::ImuConstPtr> v_imu_; ///IMU数据队列
  vector<Pose6D> IMUpose;  ///imu位姿
  vector<M3D>    v_rot_pcl_; /// 未使用
  M3D Lidar_R_wrt_IMU; ///lidar和IMU坐标系之间的旋转矩阵
  V3D Lidar_T_wrt_IMU;  ///lidar和IMU坐标系之间的平移矩阵
  V3D mean_acc; /// 求平均，加速度零偏
  V3D mean_gyr; /// 求平均，陀螺仪零偏
  V3D angvel_last; /// 上一帧角速度
  V3D acc_s_last; /// 上一帧加速度
  double start_timestamp_;  /// 开始时间戳
  double last_lidar_end_time_; /// 上一帧lidar的结束时间戳
  int    init_iter_num = 1; /// 计算陀螺仪零偏需要的IMU数据个数
  bool   b_first_frame_ = true; /// 第一帧IMU数据标志
  bool   imu_need_init_ = true; /// imu需要初始化
};


ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;    //初始化迭代次数
  Q = process_noise_cov();   /////噪声协方差的初始化
  cov_acc       = V3D(0.1, 0.1, 0.1);  //加速度测量协方差初始化
  cov_gyr       = V3D(0.1, 0.1, 0.1);  //角速度测量协方差初始化
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);  //角速度测量协方差偏置初始化
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);  //加速度测量协方差偏置初始化
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;   //上一帧角速度初始化
  Lidar_T_wrt_IMU = Zero3d;   // lidar到IMU的位置外参初始化
  Lidar_R_wrt_IMU = Eye3d;    // lidar到IMU的旋转外参初始化
  last_imu_.reset(new sensor_msgs::Imu());  //上一帧imu初始化
}

ImuProcess::~ImuProcess() {}

/// 初始化之前需要reset
void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;   //是否需要初始化imu
  start_timestamp_  = -1;     //开始时间戳
  init_iter_num     = 1;      //初始化迭代次数
  v_imu_.clear();             //imu队列清空
  IMUpose.clear();            //imu位姿清空
  last_imu_.reset(new sensor_msgs::Imu()); //上一帧imu初始化
  cur_pcl_un_.reset(new PointCloudXYZI()); //当前帧点云未去畸变初始化
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

// 初始化滤波器状态变量和协方差矩阵
// IMU初始化，需要静止
// meas：lidar和IMU数据的集合
// 计算陀螺仪零偏
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)   /// 判断是否是第一帧
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;   ///第一个时刻的加速度
    const auto &gyr_acc = meas.imu.front()->angular_velocity;  ///第一个时刻的角速度
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;  ///加速度初值作均值
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;  ///角速度初值作均值
    first_lidar_time = meas.lidar_beg_time;   /// 当前lidar时间作为初始时间
  }

  for (const auto &imu : meas.imu) 
  {
    if (N > 1) 
    {
      const auto &imu_acc = imu->linear_acceleration;
      const auto &gyr_acc = imu->angular_velocity;
      cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
      cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

      /// 迭代更新均值
      mean_acc      += (cur_acc - mean_acc) / N;
      mean_gyr      += (cur_gyr - mean_gyr) / N;
      /// Matrix.cwiseProduct()：返回两个矩阵同位置的元素分别相乘的新矩阵。
      cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) / (N - 1.0);  
      cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) / (N - 1.0);
    }
    N++;
    // cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);  
    // cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;
  }
  state_ikfom init_state = kf_state.get_x();
  // G_m_s2: 重力常数，不同地方不同
  // S2形式的单位重力向量
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;   /// 陀螺仪零偏
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;  ///lidar和IMU外参：平移矩阵
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;  ///lidar和IMU外参：旋转矩阵
  kf_state.change_x(init_state);  

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();  ///单位阵
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001; /// IMU和lidar坐标系之间的旋转角
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;   ///IMU和lidar之间的杆臂
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;  ///陀螺仪零偏
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;   ///加速度零偏
  init_P(21,21) = init_P(22,22) = 0.00001;    ///重力向量
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

}

// forward propagation at each imu point: update state
// backward propagation: compensation of point cloud distortion
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;  
  v_imu.push_front(last_imu_);  ///从队列前面插入，将上一帧最后的IMU数据插入当前帧头
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();  ///IMU帧起始时间
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();  ///IMU帧结束时间
  const double &pcl_beg_time = meas.lidar_beg_time;  ///lidar数据帧起始时间
  const double &pcl_end_time = meas.lidar_end_time;  ///lidar数据帧结束时间
  
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);  ///按时间对点云数据排序
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();  // 获取上一次ESKF估计的后验状态作为本次IMU预测的初始状态
  IMUpose.clear();
  //将初始状态加入IMUpose中,包含有时间间隔，上一帧加速度，上一帧角速度，上一帧速度，上一帧位置，上一帧旋转矩阵
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;  

  double dt = 0;

  input_ikfom in; ///加速度和角速度测量值
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);  ///当前帧IMU数据
    auto &&tail = *(it_imu + 1);  ///下一帧IMU数据
    /// 如果下一帧IMU数据在上一帧lidar结束时间之前，则不符合要求
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    /// 角速度和加速度取当前帧和下一帧的均值
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    /// 用重力向量调整加速度
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

     //如果IMU开始时刻早于上一帧lidar最晚时刻(因为将上次最后一个IMU插入到此次开头了，所以会出现一次这种情况)
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      //从上帧lidar结束时刻开始传播 计算与此次IMU结束时刻之间的时间差
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      //两次IMU测量时间间隔
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    /// 使用两次测量的均值
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);  ///前向传播

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();  
    angvel_last = angvel_avr - imu_state.bg;  //角速度减去零偏
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba); ///加速度减去零偏，从IMU系转到世界系
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];  ///加入重力矢量
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;  //后一个IMU时刻到当前帧lidar起始时刻的时间间隔
    //保存IMU预测过程的状态
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // 判断lidar结束时间是否晚于IMU，最后一个IMU时刻可能早于雷达末尾 也可能晚于雷达末尾
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  
  imu_state = kf_state.get_x();   //更新IMU状态，以便于下一帧使用
  last_imu_ = meas.imu.back();    ////保存最后一个IMU测量，以便于下一帧使用
  last_lidar_end_time_ = pcl_end_time;  /////保存这一帧最后一个雷达测量的结束时间，以便于下一帧使用

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);  ///前一帧IMU数据的旋转矩阵
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);  ///前一帧IMU数据的速度
    pos_imu<<VEC_FROM_ARRAY(head->pos);  ///前一帧IMU数据的位置
    acc_imu<<VEC_FROM_ARRAY(tail->acc);  ///后一帧IMU数据的加速度
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);  ///后一帧IMU数据的角速度
    /// 遍历一帧中的每一个lidar点
    /// curvature:相对于一帧中第一个lidar点的时间
    //点云时间需要迟于前一个IMU时刻 因为是在两个IMU时刻之间去畸变，此时默认雷达的时间戳在后一个IMU时刻之前
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;  //点到IMU开始时刻的时间间隔

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));  //点所在时刻的旋转矩阵
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);  //当前lidar点所在的世界位置-lidar帧结束时的世界位置
      //.conjugate()取旋转矩阵的转置 
      /// imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I = I^P_i 把lidar点坐标转换到IMU坐标系
      /// imu_state.rot.conjugate() * I^P_i把lidar坐标转换到世界系下
      /// imu_state.rot.conjugate() * I^P_i + T_ei = W^P_ei 运动补偿
      /// imu_state.offset_R_L_I.conjugate() * (W^P_ei-imu_state.offset_T_L_I) 再把lidar点坐标转换到lidar坐标系下
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}


/// IMU主程序
void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
