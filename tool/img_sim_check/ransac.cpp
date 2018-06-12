#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define LARGE_NUMBER 100000.0
#define RAD2DEG 57.2958

using namespace std;
using namespace cv;


// --------------------------- 3DOF

// calculates squared error from two point mapping; assumes rotation around Origin.
float sqErr_3Dof(Point2f p1, Point2f p2,
        float cos_alpha, float sin_alpha, Point2f T) {

    float x2_est = T.x + cos_alpha * p1.x - sin_alpha * p1.y;
    float y2_est = T.y + sin_alpha * p1.x + cos_alpha * p1.y;
    Point2f p2_est(x2_est, y2_est);
    Point2f dp = p2_est-p2;
    float sq_er = dp.dot(dp); // squared distance

    //cout<<dp<<endl;
    return sq_er;
}

// calculate RMSE for point-to-point metrics
float RMSE_3Dof(const vector<Point2f>& src, const vector<Point2f>& dst,
        const float* param, const bool* inliers, const Point2f center) {

    const bool all_inliers = (inliers==NULL); // handy when we run QUADRTATIC will all inliers
    unsigned int n = src.size();
    assert(n>0 && n==dst.size());

    float ang_rad = param[0];
    Point2f T(param[1], param[2]);
    float cos_alpha = cos(ang_rad);
    float sin_alpha = sin(ang_rad);

    double RMSE = 0.0;
    int ninliers = 0;
    for (unsigned int i=0; i<n; i++) {
        if (all_inliers || inliers[i]) {
            RMSE += sqErr_3Dof(src[i]-center, dst[i]-center, cos_alpha, sin_alpha, T);
            ninliers++;
        }
    }

    //cout<<"RMSE = "<<RMSE<<endl;
    if (ninliers>0)
        return sqrt(RMSE/ninliers);
    else
        return LARGE_NUMBER;
}

// Sets inliers and returns their count
inline int setInliers3Dof(const vector<Point2f>& src, const vector <Point2f>& dst,
        bool* inliers,
        const float* param,
        const float max_er,
        const Point2f center) {

    float ang_rad = param[0];
    Point2f T(param[1], param[2]);

    // set inliers
    unsigned int ninliers = 0;
    unsigned int n = src.size();
    assert(n>0 && n==dst.size());

    float cos_ang = cos(ang_rad);
    float sin_ang = sin(ang_rad);
    float max_sqErr = max_er*max_er; // comparing squared values

    if (inliers==NULL) {

        // just get the number of inliers (e.g. after QUADRATIC fit only)
        for (unsigned int i=0; i<n; i++) {

            float sqErr = sqErr_3Dof(src[i]-center, dst[i]-center, cos_ang, sin_ang, T);
            if ( sqErr < max_sqErr)
                ninliers++;
        }
    } else  {

        // get the number of inliers and set them (e.g. for RANSAC)
        for (unsigned int i=0; i<n; i++) {

            float sqErr = sqErr_3Dof(src[i]-center, dst[i]-center, cos_ang, sin_ang, T);
            if ( sqErr < max_sqErr) {
                inliers[i] = 1;
                ninliers++;
            } else {
                inliers[i] = 0;
            }
        }
    }

    return ninliers;
}

// fits 3DOF (rotation and translation in 2D) with least squares.
float fit3DofQUADRATICold(const vector<Point2f>& src, const vector<Point2f>& dst,
        float* param, const bool* inliers, const Point2f center) {

    const bool all_inliers = (inliers==NULL); // handy when we run QUADRTATIC will all inliers
    unsigned int n = src.size();
    assert(dst.size() == n);

    // count inliers
    int ninliers;
    if (all_inliers) {
        ninliers = n;
    } else {
        ninliers = 0;
        for (unsigned int i=0; i<n; i++){
            if (inliers[i])
                ninliers++;
        }
    }

    // under-dermined system
    if (ninliers<2) {
        //      param[0] = 0.0f; // ?
        //      param[1] = 0.0f;
        //      param[2] = 0.0f;
        return LARGE_NUMBER;
    }

    /*
     * x1*cosx(a)-y1*sin(a) + Tx = X1
     * x1*sin(a)+y1*cos(a) + Ty = Y1
     *
     * approximation for small angle a (radians) sin(a)=a, cos(a)=1;
     *
     * x1*1 - y1*a + Tx = X1
     * x1*a + y1*1 + Ty = Y1
     *
     * in matrix form M1*h=M2
     *
     *  2n x 4       4 x 1   2n x 1
     *
     * -y1 1 0 x1  *   a   =  X1
     *  x1 0 1 y1      Tx     Y1
     *                 Ty
     *                 1=Z
     *  ----------------------------
     *  src1         res      src2
     */

    //  4 x 1
    float res_ar[4]; // alpha, Tx, Ty, 1
    Mat res(4, 1, CV_32F, res_ar); // 4 x 1

    // 2n x 4
    Mat src1(2*ninliers, 4, CV_32F); // 2n x 4

    // 2n x 1
    Mat src2(2*ninliers, 1, CV_32F); // 2n x 1: [X1, Y1, X2, Y2, X3, Y3]'

    for (unsigned int i=0, row_cnt = 0; i<n; i++) {

        // use inliers only
        if (all_inliers || inliers[i]) {

            float x = src[i].x - center.x;
            float y = src[i].y - center.y;

            // first row

            // src1
            float* rowPtr = src1.ptr<float>(row_cnt);
            rowPtr[0] = -y;
            rowPtr[1] = 1.0f;
            rowPtr[2] = 0.0f;
            rowPtr[3] = x;

            // src2
            src2.at<float> (0, row_cnt) = dst[i].x - center.x;

            // second row
            row_cnt++;

            // src1
            rowPtr = src1.ptr<float>(row_cnt);
            rowPtr[0] = x;
            rowPtr[1] = 0.0f;
            rowPtr[2] = 1.0f;
            rowPtr[3] = y;

            // src2
            src2.at<float> (0, row_cnt) = dst[i].y - center.y;
        }
    }

    cv::solve(src1, src2, res, DECOMP_SVD);

    // estimators
    float alpha_est;
    Point2f T_est;

    // original
    alpha_est = res.at<float>(0, 0);
    T_est = Point2f(res.at<float>(1, 0), res.at<float>(2, 0));

    float Z = res.at<float>(3, 0);
    if (abs(Z-1.0) > 0.1) {
        //cout<<"Bad Z in fit3DOF(), Z should be close to 1.0 = "<<Z<<endl;
        //return LARGE_NUMBER;
    }
    param[0] = alpha_est; // rad
    param[1] = T_est.x;
    param[2] = T_est.y;

    // calculate RMSE
    float RMSE = RMSE_3Dof(src, dst, param, inliers, center);
    return RMSE;
} // fit3DofQUADRATICOLd()

// fits 3DOF (rotation and translation in 2D) with least squares.
float fit3DofQUADRATIC(const vector<Point2f>& src_, const vector<Point2f>& dst_,
        float* param, const bool* inliers, const Point2f center) {

    const bool debug = false;                   // print more debug info
    const bool all_inliers = (inliers==NULL);   // handy when we run QUADRTATIC will all inliers
    assert(dst_.size() == src_.size());
    int N = src_.size();

    // collect inliers
    vector<Point2f> src, dst;
    int ninliers;
    if (all_inliers) {
        ninliers = N;
        src = src_; // copy constructor
        dst = dst_;
    } else {
        ninliers = 0;
        for (int i=0; i<N; i++){
            if (inliers[i]) {
                ninliers++;
                src.push_back(src_[i]);
                dst.push_back(dst_[i]);
            }
        }
    }
    if (ninliers<2) {
        param[0] = 0.0f; // default return when there is not enough points
        param[1] = 0.0f;
        param[2] = 0.0f;
        return LARGE_NUMBER;
    }

    /* Algorithm: Least-Square Rigid Motion Using SVD by Olga Sorkine
     * http://igl.ethz.ch/projects/ARAP/svd_rot.pdf
     *
     * Subtract centroids, calculate SVD(cov),
     * R = V[1, det(VU')]'U', T = mean_q-R*mean_p
     */

    // Calculate data centroids
    Scalar centroid_src = mean(src);
    Scalar centroid_dst = mean(dst);
    Point2f center_src(centroid_src[0], centroid_src[1]);
    Point2f center_dst(centroid_dst[0], centroid_dst[1]);
    if (debug)
        cout<<"Centers: "<<center_src<<", "<<center_dst<<endl;

    // subtract centroids from data
    for (int i=0; i<ninliers; i++) {
        src[i] -= center_src;
        dst[i] -= center_dst;
    }

    // compute a covariance matrix
    float Cxx = 0.0, Cxy = 0.0, Cyx = 0.0, Cyy = 0.0;
    for (int i=0; i<ninliers; i++) {
        Cxx += src[i].x*dst[i].x;
        Cxy += src[i].x*dst[i].y;
        Cyx += src[i].y*dst[i].x;
        Cyy += src[i].y*dst[i].y;
    }
    Mat Mcov = (Mat_<float>(2, 2)<<Cxx, Cxy, Cyx, Cyy);
    Mcov /= (ninliers-1);
    if (debug)
        cout<<"Covariance-like Matrix "<<Mcov<<endl;

    // SVD of covariance
    cv::SVD svd;
    svd = SVD(Mcov, SVD::FULL_UV);
    if (debug) {
        cout<<"U = "<<svd.u<<endl;
        cout<<"W = "<<svd.w<<endl;
        cout<<"V transposed = "<<svd.vt<<endl;
    }

    // rotation (V*Ut)
    Mat V = svd.vt.t();
    Mat Ut = svd.u.t();
    float det_VUt = determinant(V*Ut);
    Mat W = (Mat_<float>(2, 2)<<1.0, 0.0, 0.0, det_VUt);
    float rot[4];
    Mat R_est(2, 2, CV_32F, rot);
    R_est = V*W*Ut;
    if (debug)
        cout<<"Rotation matrix: "<<R_est<<endl;

    float cos_est = rot[0];
    float sin_est = rot[2];
    float ang = atan2(sin_est, cos_est);

    // translation (mean_dst - R*mean_src)
    Point2f center_srcRot = Point2f(
            cos_est*center_src.x - sin_est*center_src.y,
            sin_est*center_src.x + cos_est*center_src.y);
    Point2f T_est = center_dst - center_srcRot;

    // Final estimate msg
    if (debug)
        cout<<"Estimate = "<<ang*RAD2DEG<<"deg., T = "<<T_est<<endl;

    param[0] = ang; // rad
    param[1] = T_est.x;
    param[2] = T_est.y;

    // calculate RMSE
    float RMSE = RMSE_3Dof(src_, dst_, param, inliers, center);
    return RMSE;
} // fit3DofQUADRATIC()

// RANSAC fit in 3DOF: 1D rot and 2D translation (maximizes the number of inliers)
// NOTE: no data normalization is currently performed
float fit3DofRANSAC(const vector<Point2f>& src, const vector<Point2f>& dst,
        float* best_param,  bool* inliers,
        const Point2f center ,
        const float inlierMaxEr,
        const int niter) {

    const int ITERATION_TO_SETTLE = 2;    // iterations to settle inliers and param
    const float INLIERS_RATIO_OK = 0.95f;  // stopping criterion

    // size of data vector
    unsigned int N = src.size();
    assert(N==dst.size());

    // unrealistic case
    if(N<2) {
        best_param[0] = 0.0f; // ?
        best_param[1] = 0.0f;
        best_param[2] = 0.0f;
        return LARGE_NUMBER;
    }

    unsigned int ninliers;                 // current number of inliers
    unsigned int best_ninliers = 0;     // number of inliers
    float best_rmse = LARGE_NUMBER;         // error
    float cur_rmse;                         // current distance error
    float param[3];                         // rad, Tx, Ty
    vector <Point2f> src_2pt(2), dst_2pt(2);// min set of 2 points (1 correspondence generates 2 equations)
    srand (time(NULL));

    // iterations
    for (int iter = 0; iter<niter; iter++) {

#ifdef DEBUG_RANSAC
        cout<<"iteration "<<iter<<": ";
#endif

        // 1. Select a random set of 2 points (not obligatory inliers but valid)
        int i1, i2;
        i1 = rand() % N; // [0, N[
        i2 = i1;
        while (i2==i1) {
            i2 = rand() % N;
        }
        src_2pt[0] = src[i1]; // corresponding points
        src_2pt[1] = src[i2];
        dst_2pt[0] = dst[i1];
        dst_2pt[1] = dst[i2];
        bool two_inliers[] = {true, true};

        // 2. Quadratic fit for 2 points
        cur_rmse = fit3DofQUADRATIC(src_2pt, dst_2pt, param, two_inliers, center);

        // 3. Recalculate to settle params and inliers using a larger set
        for (int iter2=0; iter2<ITERATION_TO_SETTLE; iter2++) {
            ninliers = setInliers3Dof(src, dst, inliers, param, inlierMaxEr, center);   // changes inliers
            cur_rmse = fit3DofQUADRATIC(src, dst, param, inliers, center);              // changes cur_param
        }

        // potential ill-condition or large error
        if (ninliers<2) {
#ifdef DEBUG_RANSAC
            cout<<" !!! less than 2 inliers "<<endl;
#endif
            continue;
        } else {
#ifdef DEBUG_RANSAC
            cout<<" "<<ninliers<<" inliers; ";
#endif
        }


#ifdef DEBUG_RANSAC
        cout<<"; recalculate: RMSE = "<<cur_rmse<<", "<<ninliers <<" inliers";
#endif


        // 4. found a better solution?
        if (ninliers > best_ninliers) {
            best_ninliers = ninliers;
            best_param[0] = param[0];
            best_param[1] = param[1];
            best_param[2] = param[2];
            best_rmse = cur_rmse;


#ifdef DEBUG_RANSAC
            cout<<" --- Solution improved: "<<
                    best_param[0]<<", "<<best_param[1]<<", "<<param[2]<<endl;
#endif
            // exit condition
            float inlier_ratio = (float)best_ninliers/N;
            if (inlier_ratio > INLIERS_RATIO_OK) {
#ifdef DEBUG_RANSAC
                cout<<"Breaking early after "<< iter+1<<
                        " iterations; inlier ratio = "<<inlier_ratio<<endl;
#endif
                break;
            }
        } else {
#ifdef DEBUG_RANSAC
            cout<<endl;
#endif
        }


    } // iterations

    // 5. recreate inliers for the best parameters
    ninliers = setInliers3Dof(src, dst, inliers, best_param, inlierMaxEr, center);

    return best_rmse;
} // fit3DofRANSAC()