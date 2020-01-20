#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initially, set to false till we receive our first measurement, after this, always true in a loop
  is_initialized_ = false;
  
  // State vectors Dimensions
  n_x_ = 5;
  x_ = VectorXd(n_x_);

  /*For normal vehicles, average acceleration, acceleration rate is between 3-4 m/sec2
    and maximum usually goes to about 6-7 m/sec2.
    In our model, we choose half of maximum possible value, i.e. 3 m/sec2
    Similarly, for angular acceleration, we choose 1.5 rad/sec2.

    Initial values set 30.0 and 30.0, which are obviously not perfect for our UKF
    */

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

   // Augmented state dimension
   n_aug_ = n_x_ + 2;
   // design parameter lambda
   lambda_ = 3 - n_aug_;
   //initial timestamp

   // Initialize weights here since they are common to lidar and radar
   weights_ = VectorXd(2 * n_aug_ + 1);
   weights_(0) = lambda_/(lambda_ + n_aug_);  //first weight
   for (int i = 1; i < 2*n_aug_+1; ++i)
   {
     weights_(i) = 0.5/(lambda_ + n_aug_);
   }

   //create augmented sigma points matrix
   Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
   
   time_us_ = 0.0;

  // initializing the covariance matrix P, built on identity matrix
  P_ = MatrixXd(n_x_, n_x_);

  // these values will be changed, to ensure RSME is lowest...
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0.1, 0,
        0, 0, 0, 0, 0.1;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  
   // this calulates the elapsed time between measurements
   double dt = (meas_package.timestamp_ - time_us_)/1000000.0;
   // new timestamp
   time_us_ = meas_package.timestamp_;

  // only for first measurement
  if (!is_initialized_)
  {
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    {
      // define measurements for radar
      // lidar reports values in polar coordinate system, hence we have to convert it to rectangular coordinate system

      double rho, phi, rho_dot;		//get these from measurements
      rho = meas_package.raw_measurements_[0];
      phi = meas_package.raw_measurements_[1];
      rho_dot = meas_package.raw_measurements_[2];
      
      //convert measurements from polar to cartesian
      double x, y, vx, vy, v;
      x = rho * cos(phi);
      y = rho * sin(phi);
      vx = rho_dot * cos(phi);
      vy = rho_dot * sin(phi);
      v = sqrt(vx*vx + vy*vy);
      
      // assign the measurement vector
      x_ << x, y, v, vx, vy;

    }
    else	//measurement is from lidar
    {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    
    // now we have initialized the state vector, hence the if loop will be avoided in next frame
    is_initialized_ = true;

    return;
  }
  
  // execute the prediction step now
  Prediction(dt);

  // measurement update step depends on the sensor type
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
  {
    UpdateRadar(meas_package);
  }

  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {
    UpdateLidar(meas_package);
  }

}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */

  /*
  // Not necessary
  // Generate Sigma Points
  MatrixXd Xsig = MatrixXd(n_x_, 2*n_x_+1);

  // calculate square root of matrix P
  MatrixXd A = P_.llt().matrixL();

  // set first column of sigma point matrix
  Xsig.col(0) = x_;

  // set remaining sigma points
  for (int i = 0; i < n_x_; ++i)
  {
    Xsig.col(i+1)     = x_ + sqrt(lambda_+n_x_) * A.col(i);
    Xsig.col(i+1+n_x_) = x_ - sqrt(lambda_+n_x_) * A.col(i);
  }
  */

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
 
  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;         // linear acceleration covariance
  P_aug(6,6) = std_yawdd_*std_yawdd_; // angular acceleration covariance

  // create square root matrix
  
  MatrixXd sqrt_P_aug = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  
  for (int i = 0; i < n_aug_; ++i)
  {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * sqrt_P_aug.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * sqrt_P_aug.col(i);
  }

  // predict sigma points
  for (int i = 0; i < 2*n_aug_+1; ++i)
  {
    // initialize all variables
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yaw_rate = Xsig_aug(4, i);
    double acc_a = Xsig_aug(5, i);
    double acc_yawdd = Xsig_aug(6, i);
      
    // predicted state values
    double px_p, py_p;

    //avoiding division by zero
    if (fabs(yaw_rate) > 0.001)
    {
      px_p = p_x + (v/yaw_rate)*(sin(yaw+yaw_rate*delta_t) - sin(yaw));
      py_p = p_y + (v/yaw_rate)*(-cos(yaw+yaw_rate*delta_t) + cos(yaw));
    }
    else
    {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v *delta_t * sin(yaw);
    }
      
    double v_p = v;
    double yaw_p = yaw + yaw_rate * delta_t;
    double yaw_rate_p = yaw_rate;
      
    // add noise
    px_p = px_p + 0.5 * delta_t * delta_t * cos(yaw) * acc_a;
    py_p = py_p + 0.5 * delta_t * delta_t * sin(yaw) * acc_a;
    v_p = v_p + acc_a * delta_t;
    yaw_p = yaw_p + 0.5 * delta_t * delta_t * acc_yawdd;
    yaw_rate_p = yaw_rate_p + 0 + delta_t * acc_yawdd;
      
    // write predicted sigma points into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yaw_rate_p;

  }

  // Predict state mean
  x_.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; ++i)
  {
    x_ = x_ + weights_(i)*Xsig_pred_.col(i);
  }

  // Predict state covairance
  P_.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; ++i)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while(x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
    while(x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;
    P_ = P_ + weights_(i)*x_diff*x_diff.transpose();
  }

}



void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

   // now we have to get the latest measurement from lidar
   VectorXd z = meas_package.raw_measurements_;
   //setting measurement dimension
   int n_z = 2;
   MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);
   
   // update measurement
   for(int i = 0; i < 2*n_aug_+1; ++i)
   {
     Zsig(0, i) = Xsig_pred_(0, i);
     Zsig(1, i) = Xsig_pred_(1, i);
   }

   //Predicted mean measurement
   VectorXd z_pred = VectorXd(n_z);
   z_pred.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; ++i)
   {
     z_pred = z_pred + weights_(i) * Zsig.col(i);
   }

   
   // calculate covariance of predicted measurement
   MatrixXd S = MatrixXd(n_z, n_z);
   S.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; ++i)
   {
     VectorXd z_diff = Zsig.col(i) - z_pred;

     S = S + weights_(i) * z_diff * z_diff.transpose();
   }
   
   //  defining noise covariance matrix for lidar here
   MatrixXd R_lidar_ = MatrixXd(2,2);
   R_lidar_ << std_laspx_*std_laspx_, 0,
               0, std_laspy_*std_laspy_;
  
   // adding noise
   S = S + R_lidar_;
   
   // Update UKF step
   // create matrix for cross correlation Tc
   MatrixXd Tc = MatrixXd(n_x_, n_z);
   Tc.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; ++i)
   {
     // predicted state difference
     VectorXd x_diff = Xsig_pred_.col(i) - x_;

     // predicted measurement difference
     VectorXd z_diff = Zsig.col(i) - z_pred;

     Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
   }
   
   // residual (deviation from actual measurements)
   VectorXd z_diff = z - z_pred;
   
   // calculate Kalman gain K;
   MatrixXd K = Tc * S.inverse();
   
   // update state mean
   x_ = x_ + K * z_diff;

   // update covariance
   P_ = P_ - K * S * K.transpose();

   // ------------------------- This part is for NIS calculations --------------------------//
  
   double NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
   //std::cout<<"NIS_laser = "<<NIS_laser_<<std::endl;

}


void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
   // Measurement prediction
   // set measurement dimension, radar can measure r, phi, and r_dot
   
   int n_z = 3; //dimensions of radar measurement

   //extract measurement from radar
   VectorXd z = meas_package.raw_measurements_;

   MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);

   for(int i = 0; i < 2 * n_aug_ + 1; ++i)
   {
     
     double p_x = Xsig_pred_(0, i);
     double p_y = Xsig_pred_(1, i);
     double v = Xsig_pred_(2, i);
     double yaw = Xsig_pred_(3, i);
     double yawd = Xsig_pred_(4, i);
     
     //transform points from cartesian to polar
     Zsig(0, i) = sqrt(p_x*p_x+p_y*p_y);
     Zsig(1, i) = atan2(p_y, p_x);
     Zsig(2, i) = (p_x*v*cos(yaw)+p_y*v*sin(yaw))/(sqrt(p_x*p_x+p_y*p_y));

   }
   
   // predicted mean measurement
   // create example vector for mean predicted measurement
   VectorXd z_pred = VectorXd(n_z);
   
   z_pred.fill(0.0);
   for (int i = 0; i < 2*n_aug_+1; ++i)
   {
     z_pred = z_pred + weights_(i) * Zsig.col(i);
   }
   
   // Creating predicted measurement covariance matrix
   MatrixXd S = MatrixXd(n_z,n_z);
   S.fill(0.0);
   for (int i = 0; i < 2*n_aug_+1; ++i)
   {
     VectorXd z_diff = Zsig.col(i) - z_pred;
     S = S + weights_(i) * z_diff * z_diff.transpose();
   }
   
   // defining noise covariance matrix for radar here
   MatrixXd R_radar_ = MatrixXd(3, 3);
   R_radar_ << std_radr_*std_radr_, 0, 0,
               0, std_radphi_*std_radphi_, 0,
               0, 0, std_radrd_*std_radrd_;
               
    //adding noise
    S = S + R_radar_;

   // UKF update
   // create matrix for cross correlation Tc

   MatrixXd Tc = MatrixXd(n_x_, n_z);
   Tc.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; ++i)
   {
     // predicted state difference
     VectorXd x_diff = Xsig_pred_.col(i) - x_;
     // angle normalization
     while(x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
     while(x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;

     //predicted measurement difference
     VectorXd z_diff = Zsig.col(i) - z_pred;
     while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
     while(z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

     Tc = Tc + weights_(i) * x_diff * z_diff.transpose();

   }

   // residual (deviation from actual measurements)
   VectorXd z_diff = z - z_pred;
   
   // calculate Kalman gain K
   MatrixXd K = Tc * S.inverse();
   // angle normalization
   while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
   while(z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;
   
   // update state mean
   x_ = x_ + K * z_diff;

   // update covariance
   P_ = P_ - K * S * K.transpose();

   // ------------------------- This part is for NIS calculations --------------------------//
   
   double NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
   //std::cout<<"NIS_radar = "<<NIS_radar_<<std::endl;

}
