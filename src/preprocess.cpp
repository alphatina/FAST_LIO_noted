#include "preprocess.h"

#define RETURN0     0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess()
  :feature_enabled(0), lidar_type(AVIA), blind(0.01), point_filter_num(1)
{
  inf_bound = 10;  /// 距离阈值，大于10m表明lidar点太远，无效
  N_SCANS   = 6;  /// 6线lidar
  SCAN_RATE = 10;  /// 每秒10个scan
  group_size = 8;  /// 8个点为一组
  disA = 0.01;   /// 点集的距离阈值,判断是否是平面
  disB = 0.1; /// 点集的距离阈值,判断是否是平面
  p2l_ratio = 225;  // 点到线的距离阈值，需要大于这个值才能判断组成面
  limit_maxmid =6.25; /// 中点到左侧的距离变化率范围
  limit_midmin =6.25; /// 中点到右侧的距离变化率范围
  limit_maxmin = 3.24; /// 左侧到右侧的距离变化率范围
  jump_up_limit = 170.0; 
  jump_down_limit = 8.0;
  cos160 = 160.0;
  edgea = 2;  /// 点与点距离超过两倍，则认为遮挡
  edgeb = 0.1;  /// 点与点距离超过0.1米，则认为遮挡
  smallp_intersect = 172.5;
  smallp_ratio = 1.2;  
  given_offset_time = false;  //是否提供时间偏移

  jump_up_limit = cos(jump_up_limit/180*M_PI);      /// 170度的cos值
  jump_down_limit = cos(jump_down_limit/180*M_PI);  // 8度的cos值
  cos160 = cos(cos160/180*M_PI);   /// 160度的cos值
  smallp_intersect = cos(smallp_intersect/180*M_PI);  ///172.5度的cos值
}

Preprocess::~Preprocess() {}


/// @brief  该函数实际未被使用
void Preprocess::set(bool feat_en, int lid_type, double bld, int pfilt_num)
{
  feature_enabled = feat_en;  //// 是否提取特征点
  lidar_type = lid_type;
  blind = bld;  //最小距离阈值，即过滤掉0～blind范围内的点云
  point_filter_num = pfilt_num;  //采样间隔，每隔point_filter_num个点取1个点
}

void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{  
  avia_handler(msg);
  *pcl_out = pl_surf;
}

/// 默认时间单位是ms
void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  switch (time_unit)
  {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  switch (lidar_type)
  {
  case OUST64:
    oust64_handler(msg);
    break;

  case VELO16:
    velodyne_handler(msg);
    break;
  
  default:
    printf("Error LiDAR Type");
    break;
  }
  *pcl_out = pl_surf;
}

/// 预处理livox lidar数据
void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
  /// 清除之前的点云缓存
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  double t1 = omp_get_wtime();  /// 返回从某个时间点到现在过去的时间
  int plsize = msg->point_num;   /// 一帧点云的lidar点数
  // cout<<"plsie: "<<plsize<<endl;

  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  /// 每个scan都是buff中的一组点云
  for(int i=0; i<N_SCANS; i++)
  {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);
  }
  uint valid_num = 0;
  
  /*** 如果提特征
   * 先不降采样，所有点都放进pl_full，满足条件的有效点放进pl_buff
   * 点数少于5的line不处理
   * 提取的特征包括：点到lidar的水平距离，当前点到后一个点距离
   * 这些特征放进pl_buff的对应点中
   * 调用give_feature函数，筛选角点放入pl_corn，降采样后的平面点放入pl_surf
  ***/
  /*** 如果不提特征
   * 直接降采样，降采样后的所有点都放进pl_full，降采样后满足要求的点放进pl_surf
   * 降采样后不在盲区的有效点都放进pl_surf
  ***/
  if (feature_enabled)
  {
    for(uint i=1; i<plsize; i++)
    {
      // 只取线数在0~N_SCANS内并且回波次序为0或者1的点
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        pl_full[i].x = msg->points[i].x;
        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity;
        pl_full[i].curvature = msg->points[i].offset_time / float(1000000); //use curvature as time of each laser points ，单位us

        bool is_new = false;
        // 只有当前点和上一点的间距足够大（>1e-7），才将当前点认为是有用的点，将当前点加入到对应line的pl_buff队列中
        if((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) 
            || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
            || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
        {
          pl_buff[msg->points[i].line].push_back(pl_full[i]);
        }
      }
    }
    static int count = 0;
    static double time = 0.0;
    count ++;
    double t0 = omp_get_wtime();

    /// 分别处理每个scan的lidar点，scan似乎就是line
    for(int j=0; j<N_SCANS; j++)
    {
      /// 如果line中的点不超过5个，跳过，不处理
      if(pl_buff[j].size() <= 5)
        continue;

      pcl::PointCloud<PointType> &pl = pl_buff[j];
      plsize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(plsize);
      plsize--;
      for(uint i=0; i<plsize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);  /// 点到lidar的水平距离
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = sqrt(vx * vx + vy * vy + vz * vz);
      }
      types[plsize].range = sqrt(pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y);
      give_feature(pl, types);   /// 筛选角点放入pl_corn，平面点放入pl_surf
      // pl_surf += pl;
    }
    time += omp_get_wtime() - t0;
    printf("Feature extraction time: %lf \n", time / count);
  }
  else
  {
    for(uint i=1; i<plsize; i++)
    {
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        valid_num ++;

        /// 降采样
        if (valid_num % point_filter_num == 0)
        {
          pl_full[i].x = msg->points[i].x;
          pl_full[i].y = msg->points[i].y;
          pl_full[i].z = msg->points[i].z;
          pl_full[i].intensity = msg->points[i].reflectivity;
          pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // use curvature as time of each laser points

          // 只有当前点和上一点的间距足够大（>1e-7），并且在最小距离阈值之外，才将当前点认为是有用的点，加入到pl_surf队列中
          if(((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) 
              || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
              || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
              && (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > (blind * blind)))
          {
            pl_surf.push_back(pl_full[i]);
          }
        }
      }
    }
  }
}

void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  if (feature_enabled)
  {
    for (int i = 0; i < N_SCANS; i++)
    {
      pl_buff[i].clear();
      pl_buff[i].reserve(plsize);
    }

    for (uint i = 0; i < plsize; i++)
    {
      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      if (range < (blind * blind)) continue;
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.3;
      if (yaw_angle >= 180.0)
        yaw_angle -= 360.0;
      if (yaw_angle <= -180.0)
        yaw_angle += 360.0;

      added_pt.curvature = pl_orig.points[i].t * time_unit_scale;
      if(pl_orig.points[i].ring < N_SCANS)
      {
        pl_buff[pl_orig.points[i].ring].push_back(added_pt);
      }
    }

    for (int j = 0; j < N_SCANS; j++)
    {
      PointCloudXYZI &pl = pl_buff[j];
      int linesize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < linesize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = vx * vx + vy * vy + vz * vz;
      }
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
      give_feature(pl, types);
    }
  }
  else
  {
    double time_stamp = msg->header.stamp.toSec();
    // cout << "===================================" << endl;
    // printf("Pt size = %d, N_SCANS = %d\r\n", plsize, N_SCANS);
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
      if (i % point_filter_num != 0) continue;

      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      
      if (range < (blind * blind)) continue;
      
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.curvature = pl_orig.points[i].t * time_unit_scale; // curvature unit: ms

      pl_surf.points.push_back(added_pt);
    }
  }
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * SCAN_RATE;       // scan angular velocity
    std::vector<bool> is_first(N_SCANS,true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);      // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0)
    {
      given_offset_time = true;
    }
    else
    {
      given_offset_time = false;
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
      double yaw_end  = yaw_first;
      int layer_first = pl_orig.points[0].ring;
      for (uint i = plsize - 1; i > 0; i--)
      {
        if (pl_orig.points[i].ring == layer_first)
        {
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
          break;
        }
      }
    }

    if(feature_enabled)
    {
      for (int i = 0; i < N_SCANS; i++)
      {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
      }
      
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        int layer  = pl_orig.points[i].ring;
        if (layer >= N_SCANS) continue;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale; // units: ms

        if (!given_offset_time)
        {
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;
          if (is_first[layer])
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        pl_buff[layer].points.push_back(added_pt);
      }

      for (int j = 0; j < N_SCANS; j++)
      {
        PointCloudXYZI &pl = pl_buff[j];
        int linesize = pl.size();
        if (linesize < 2) continue;
        vector<orgtype> &types = typess[j];
        types.clear();
        types.resize(linesize);
        linesize--;
        for (uint i = 0; i < linesize; i++)
        {
          types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
          vx = pl[i].x - pl[i + 1].x;
          vy = pl[i].y - pl[i + 1].y;
          vz = pl[i].z - pl[i + 1].z;
          types[i].dista = vx * vx + vy * vy + vz * vz;
        }
        types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
        give_feature(pl, types);
      }
    }
    else
    {
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;
        
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

        if (!given_offset_time)
        {
          int layer = pl_orig.points[i].ring;
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

          if (is_first[layer])
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          // compute offset time
          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
          if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
          {
            pl_surf.points.push_back(added_pt);
          }
        }
      }
    }
}

void Preprocess::give_feature(pcl::PointCloud<PointType> &pl, vector<orgtype> &types)
{
  int plsize = pl.size();
  int plsize2;
  if(plsize == 0)
  {
    printf("something wrong\n");
    return;
  }
  uint head = 0;

  /// 从非盲区的点开始
  while(types[head].range < blind)
  {
    head++;
  }

  // Surf  group_size = 8，用8个点判断平面
  plsize2 = (plsize > group_size) ? (plsize - group_size) : 0;

  Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero()); /// 当前平面法向量
  Eigen::Vector3d last_direct(Eigen::Vector3d::Zero()); /// 上一平面法向量

  uint i_nex = 0, i2;  
  uint last_i = 0; /// 上一个点的索引
  uint last_i_nex = 0;  /// 上一个点的下一个点的索引
  int last_state = 0;  /// 1代表上个状态是平面，否则是0
  int plane_type;

  /// 用8个点判断平面
  for(uint i=head; i<plsize2; i++)
  {
    if(types[i].range < blind)
    {
      continue;
    }

    i2 = i; /// i2是当前点

    plane_type = plane_judge(pl, types, i, i_nex, curr_direct);
    
    if(plane_type == 1)  /// 是平面
    {
      /// 设置平面点类型
      for(uint j=i; j<=i_nex; j++)
      { 
        if(j!=i && j!=i_nex)
        {
          types[j].ftype = Real_Plane;  //把起始点和终止点之间的所有点设置为确定的平面点
        }
        else
        {
          types[j].ftype = Poss_Plane;  //把起始点和终止点设置为可能的平面点
        }
      }
      
      // if(last_state==1 && fabs(last_direct.sum())>0.5)
      //如果之前状态是平面，判断当前点是处于两平面边缘的点还是平面中的点
      if(last_state==1 && last_direct.norm()>0.1)
      {
        double mod = last_direct.transpose() * curr_direct;
        if(mod>-0.707 && mod<0.707)  /// 两个平面的法向量夹角大于45度
        {
          types[i].ftype = Edge_Plane;
        }
        else
        {
          types[i].ftype = Real_Plane;
        }
      }
      
      i = i_nex - 1;
      last_state = 1;
    }
    else // if(plane_type == 2)
    {
      i = i_nex;
      last_state = 0;
    }
    // else if(plane_type == 0)
    // {
    //   if(last_state == 1)
    //   {
    //     uint i_nex_tem;
    //     uint j;
    //     for(j=last_i+1; j<=last_i_nex; j++)
    //     {
    //       uint i_nex_tem2 = i_nex_tem;
    //       Eigen::Vector3d curr_direct2;

    //       uint ttem = plane_judge(pl, types, j, i_nex_tem, curr_direct2);

    //       if(ttem != 1)
    //       {
    //         i_nex_tem = i_nex_tem2;
    //         break;
    //       }
    //       curr_direct = curr_direct2;
    //     }

    //     if(j == last_i+1)
    //     {
    //       last_state = 0;
    //     }
    //     else
    //     {
    //       for(uint k=last_i_nex; k<=i_nex_tem; k++)
    //       {
    //         if(k != i_nex_tem)
    //         {
    //           types[k].ftype = Real_Plane;
    //         }
    //         else
    //         {
    //           types[k].ftype = Poss_Plane;
    //         }
    //       }
    //       i = i_nex_tem-1;
    //       i_nex = i_nex_tem;
    //       i2 = j-1;
    //       last_state = 1;
    //     }

    //   }
    // }

    last_i = i2;
    last_i_nex = i_nex;
    last_direct = curr_direct;
  }

  // 用3个点判断直线
  plsize2 = plsize > 3 ? plsize - 3 : 0;
  for(uint i=head+3; i<plsize2; i++)
  {
    if(types[i].range<blind || types[i].ftype>=Real_Plane)
    {
      continue;
    }

    if(types[i-1].dista<1e-16 || types[i].dista<1e-16)
    {
      continue;
    }

    Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);  ///当前点
    Eigen::Vector3d vecs[2];

    for(int j=0; j<2; j++)
    {
      int m = -1;
      if(j == 1)
      {
        m = 1;
      }

      
      if(types[i+m].range < blind) /// 若当前的前/后一个点在盲区内 (<0.01 m)
      {
        if(types[i].range > inf_bound)  /// 如果当前点过远（> 10 m)
        {
          types[i].edj[j] = Nr_inf;
        }
        else
        {
          types[i].edj[j] = Nr_blind;  /// 靠近盲区
        }
        continue;
      }

      vecs[j] = Eigen::Vector3d(pl[i+m].x, pl[i+m].y, pl[i+m].z);  ///前后点坐标
      vecs[j] = vecs[j] - vec_a;  /// 前后点到当前点的向量
      
      //若lidar坐标系原点为O 当前点为A 前/后点为M和N
      //则下面OA点乘MA/（|OA|*|MA|）
      //得到的是角OAM/OAN的cos
      types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm();
      if(types[i].angle[j] < jump_up_limit)
      {
        types[i].edj[j] = Nr_180;  /// M在OA延长线上
      }
      else if(types[i].angle[j] > jump_down_limit)
      {
        types[i].edj[j] = Nr_zero;  /// M在OA延长线上
      }
    }

    types[i].intersect = vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm();

    /// 如果前点是正常点，后点在当前点的激光线上，当前点到后点距离大于2.25 cm，且大于当前点到前点的距离的4倍
    if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_zero && types[i].dista>0.0225 && types[i].dista>4*types[i-1].dista)
    {
      if(types[i].intersect > cos160)  /// 角MAN小于160度，前点不在激光线上
      {
        if(edge_jump_judge(pl, types, i, Prev))
        {
          types[i].ftype = Edge_Jump;  ///跳变的边缘点
        }
      }
    }
    else if(types[i].edj[Prev]==Nr_zero && types[i].edj[Next]== Nr_nor && types[i-1].dista>0.0225 && types[i-1].dista>4*types[i].dista)  /// 如果前点在激光线上，后点是正常点，当前点到前点距离大于2.25 cm，且大于当前点到后点的距离的4倍
    {
      if(types[i].intersect > cos160)  
      {
        if(edge_jump_judge(pl, types, i, Next))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_inf) /// 如果前点是正常点，后点距离过远
    {
      if(edge_jump_judge(pl, types, i, Prev))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    else if(types[i].edj[Prev]==Nr_inf && types[i].edj[Next]==Nr_nor) /// 如果前点距离过远，后点是正常点
    {
      if(edge_jump_judge(pl, types, i, Next))
      {
        types[i].ftype = Edge_Jump;
      }
     
    }
    else if(types[i].edj[Prev]>Nr_nor && types[i].edj[Next]>Nr_nor)  /// 如果前后点都是正常点
    {
      if(types[i].ftype == Nor)   /// 如果当前点未判定
      {
        types[i].ftype = Wire;   /// 小线段的中间点，无用
      }
    }
  }

  plsize2 = plsize-1;
  double ratio;
  for(uint i=head+1; i<plsize2; i++)
  {
    if(types[i].range<blind || types[i-1].range<blind || types[i+1].range<blind)
    {
      continue;
    }
    
    if(types[i-1].dista<1e-8 || types[i].dista<1e-8)
    {
      continue;
    }

    if(types[i].ftype == Nor)   /// 如果当前点还未判定
    {
      if(types[i-1].dista > types[i].dista)
      {
        ratio = types[i-1].dista / types[i].dista;
      }
      else
      {
        ratio = types[i].dista / types[i-1].dista;
      }

      /// 如果夹角MAN大于172.5度，且MA和MN长度相似（比例小于1.2），则当前点是平面点
      if(types[i].intersect<smallp_intersect && ratio < smallp_ratio)
      {
        if(types[i-1].ftype == Nor)  
        {
          types[i-1].ftype = Real_Plane;
        }
        if(types[i+1].ftype == Nor)
        {
          types[i+1].ftype = Real_Plane;
        }
        types[i].ftype = Real_Plane;  
      }
    }
  }


  /// 把平面点降采样后放进pl_surf
  int last_surface = -1;
  for(uint j=head; j<plsize; j++)
  {
    if(types[j].ftype==Poss_Plane || types[j].ftype==Real_Plane)
    {
      if(last_surface == -1)
      {
        last_surface = j;
      }

      /// 降采样存储
      if(j == uint(last_surface+point_filter_num-1))
      {
        PointType ap;
        ap.x = pl[j].x;
        ap.y = pl[j].y;
        ap.z = pl[j].z;
        ap.intensity = pl[j].intensity;
        ap.curvature = pl[j].curvature;
        pl_surf.push_back(ap);

        last_surface = -1;
      }
    }
    else
    {
      /// 如果在边缘，就是角点
      if(types[j].ftype==Edge_Jump || types[j].ftype==Edge_Plane)
      {
        pl_corn.push_back(pl[j]);
      }
      //假如上个平面点被无差别滤波掉了，而此时已经到了边缘
      //取上个平面点到此次边缘线之间的所有点的重心当作一个平面点存储进去
      if(last_surface != -1)
      {
        PointType ap;
        for(uint k=last_surface; k<j; k++)
        {
          ap.x += pl[k].x;
          ap.y += pl[k].y;
          ap.z += pl[k].z;
          ap.intensity += pl[k].intensity;
          ap.curvature += pl[k].curvature;
        }
        ap.x /= (j-last_surface);
        ap.y /= (j-last_surface);
        ap.z /= (j-last_surface);
        ap.intensity /= (j-last_surface);
        ap.curvature /= (j-last_surface);
        pl_surf.push_back(ap);
      }
      last_surface = -1;
    }
  }
}

void Preprocess::pub_func(PointCloudXYZI &pl, const ros::Time &ct)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "livox";
  output.header.stamp = ct;
}

int Preprocess::plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct)
{
  double group_dis = disA*types[i_cur].range + disB;  ///0.01*sqrt(x^2+y^2)+0.1 基本上可以近似看成是0.1 100m的时候才到0.2
  group_dis = group_dis * group_dis;
  // i_nex = i_cur;

  double two_dis;
  vector<double> disarr;  /// 前后点距离数组
  disarr.reserve(20);

  /// 取8个点的距离
  for(i_nex=i_cur; i_nex<i_cur+group_size; i_nex++)
  {
    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();  /// 附近的点距离lidar太近了，将法向量设置为零向量
      return 2;  /// 非平面点
    }
    disarr.push_back(types[i_nex].dista);   /// 存储当前点和后点之间的距离
  }
  
  //看看后续的点有没有满足条件的
  for(;;)
  {
    if((i_cur >= pl.size()) || (i_nex >= pl.size())) break;

    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }

    //i_cur点到最后的i_nex点的向量
    vx = pl[i_nex].x - pl[i_cur].x;
    vy = pl[i_nex].y - pl[i_cur].y;
    vz = pl[i_nex].z - pl[i_cur].z;
    two_dis = vx*vx + vy*vy + vz*vz;  /// 距离平方
    if(two_dis >= group_dis)   /// 距离当前点太远了，直接break，后面的点不必再看
    {
      break;
    }
    disarr.push_back(types[i_nex].dista);
    i_nex++;
  }

  double leng_wid = 0;
  double v1[3], v2[3];
  // 假设i_cur点为A，j点为B，i_nex点为C
  // i_nex是i_cur的最后一个近邻点
  for(uint j=i_cur+1; j<i_nex; j++)
  {
    if((j >= pl.size()) || (i_cur >= pl.size())) break;
    // 向量AB
    v1[0] = pl[j].x - pl[i_cur].x;
    v1[1] = pl[j].y - pl[i_cur].y;
    v1[2] = pl[j].z - pl[i_cur].z;

    //向量AB叉乘向量AC
    v2[0] = v1[1]*vz - vy*v1[2];
    v2[1] = v1[2]*vx - v1[0]*vz;
    v2[2] = v1[0]*vy - vx*v1[1];

    //ABC组成的平行四边形面积的平方 = (|AC|*h)^2, h是B点到AC的距离
    double lw = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
    if(lw > leng_wid)
    {
      leng_wid = lw;  ///寻找最大面积的平方，也就是寻找距离AC最远的B
    }
  }

  //|AC|*|AC|/(h*h)<225，也就是h>|AC|/15，B点到AC的距离大于AC边长的1/15，这种情况下当前点非平面点
  if((two_dis*two_dis/leng_wid) < p2l_ratio)
  {
    curr_direct.setZero();
    return 0;
  }

  //把两点之间的距离 从大到小排序
  uint disarrsize = disarr.size();
  for(uint j=0; j<disarrsize-1; j++)
  {
    for(uint k=j+1; k<disarrsize; k++)
    {
      if(disarr[j] < disarr[k])
      {
        leng_wid = disarr[j];
        disarr[j] = disarr[k];
        disarr[k] = leng_wid;
      }
    }
  }

  /// 如果有邻近点之间的距离太小，则法向量为零向量，当前点非平面点
  if(disarr[disarr.size()-2] < 1e-16)
  {
    curr_direct.setZero();
    return 0;
  }

  if(lidar_type==AVIA)
  {

    double dismax_mid = disarr[0]/disarr[disarrsize/2];
    double dismid_min = disarr[disarrsize/2]/disarr[disarrsize-2];
    ///点与点之间距离变化太大的时候 可能与激光束是平行的 就也舍弃了
    if(dismax_mid>=limit_maxmid || dismid_min>=limit_midmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  else
  {
    double dismax_min = disarr[0] / disarr[disarrsize-2];
    if(dismax_min >= limit_maxmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  
  //法向量是当前点指向最后的i_nex点的向量
  curr_direct << vx, vy, vz;
  curr_direct.normalize();
  return 1; /// 是平面点
}

bool Preprocess::edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir)
{
  if(nor_dir == 0)  /// prev
  {
    if(types[i-1].range<blind || types[i-2].range<blind)   /// 前两个点在盲区则该点不是边缘点
    {
      return false;
    }
  }
  else if(nor_dir == 1)  /// next
  {
    if(types[i+1].range<blind || types[i+2].range<blind)  /// 后两个点在盲区则该点不是边缘点
    {
      return false;
    }
  }

  //// 如果nor_dir = prev = 0; 则d1是i和前点的距离，d2是i的前点和前前点的距离
  //// 如果nor_dir = next = 1; 则d1是i和后点的距离，d2是i的后点和后后点的距离
  double d1 = types[i+nor_dir-1].dista;   ///该点和
  double d2 = types[i+3*nor_dir-2].dista;
  double d;

  /// 排序，d1>d2
  if(d1<d2)
  {
    d = d1;
    d1 = d2;
    d2 = d;
  }

  d1 = sqrt(d1);
  d2 = sqrt(d2);
  
  //假如连续点的间距差太大，则后面可能是被遮挡，就不把它当作边缘点
  if(d1>edgea*d2 || (d1-d2)>edgeb)
  {
    return false;
  }
  
  return true;
}
