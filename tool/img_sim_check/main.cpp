#include <iostream>
#include <stdio.h>
#include <vector>
#include "DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "DescManip.h"

bool debug_mode=false;

bool check_geo_insistence(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, std::vector<cv::DMatch>& good_matches, float img_size){
    std::vector<float> scale_rate_hist;
    std::vector<float> rot_hist;
    std::vector<float> len_hist;
    std::vector<cv::Point2f> kp1_f;
    std::vector<cv::Point2f> kp2_f;
    for(int i=0; i<good_matches.size();i++){
        int id1 = good_matches[i].queryIdx;
        int id2 = good_matches[i].trainIdx;
        if(kp1.size()<=id1){
            std::cout<<"kp1[id1] overflow!!!"<<std::endl;
        }
        if(kp2.size()<=id2){
            std::cout<<"kp2[id2] overflow!!!"<<std::endl;
        }
        kp1_f.push_back(kp1[id1].pt);
        kp2_f.push_back(kp2[id2].pt);
    }
    for(int i=0; i<kp1_f.size();i++){
        for(int j=i+1; j<kp1_f.size();j++){
            cv::Point2f ray1=kp1_f[j]-kp1_f[i];
            cv::Point2f ray2=kp2_f[j]-kp2_f[i];
            float dis = cv::norm(ray1);
            float dis2 = cv::norm(ray2);
            float ang=1;
            if (dis >10 && dis2 >10){
                ray1=ray1/dis;
                ray2=ray2/dis2;
                ang=ray1.x*ray2.x+ray1.y*ray2.y;
            }
            rot_hist.push_back(ang);
            if (dis2 >10){
                scale_rate_hist.push_back(dis/dis2);
            }
            
            len_hist.push_back(dis);
        }
    }
    if (rot_hist.size()<100 ||scale_rate_hist.size()<100){
        if(debug_mode){
            std::cout<<"too few data: "<<rot_hist.size()<<std::endl;
            std::cout<<"too few data: "<<scale_rate_hist.size()<<std::endl;
        }
        return false;
    }
    int avg_len=0;
    for (int i=0; i<len_hist.size();i++){
        avg_len = avg_len+len_hist[i];
    }
    avg_len=avg_len/len_hist.size();
    float rot_avg=0;
    for (int i=0; i<rot_hist.size();i++){
        rot_avg = rot_avg+rot_hist[i];
    }
    rot_avg=rot_avg/rot_hist.size();
    int rot_error_cont=0;
    for (int i=0; i<rot_hist.size();i++){
        if(fabs(rot_hist[i]-rot_avg)>rot_avg*0.3){
            rot_error_cont++;
        }
    }
    float scale_avg=0;
    for (int i=0; i<scale_rate_hist.size();i++){
        scale_avg = scale_avg+scale_rate_hist[i];
    }
    scale_avg=scale_avg/scale_rate_hist.size();
    int scale_error_cont=0;
    for (int i=0; i<scale_rate_hist.size();i++){
        if(fabs(scale_rate_hist[i]-scale_avg)>scale_avg*0.3){
            scale_error_cont++;
        }
    }
    if(debug_mode){
        std::cout<<"len avg: "<<avg_len<<std::endl;
        std::cout<<"len rate: "<<(float)avg_len/img_size<<std::endl;
        std::cout<<"scale_avg: "<<scale_avg<<std::endl;
        std::cout<<"scale_error_cont: "<<scale_error_cont<<std::endl;
        std::cout<<"scale_rate_hist.size(): "<<scale_rate_hist.size()<<std::endl;
        std::cout<<"rot_avg: "<<rot_avg<<std::endl;
        std::cout<<"rot_error_cont: "<<rot_error_cont<<std::endl;
        std::cout<<"rot_hist.size(): "<<rot_hist.size()<<std::endl;
    }
    if ((float)avg_len/img_size<0.1){
        return false;
    }
    if(rot_error_cont>rot_hist.size()*0.3){
        return false;
    }
    if(scale_error_cont>scale_rate_hist.size()*0.3){
        return false;
    }
    return true;
}

std::vector<cv::Mat> loadFeatures( std::vector<std::string>& path_to_images,std::string descriptor, std::vector<std::vector<cv::KeyPoint> >& kps_list, bool bFlip=false) throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create();
    else if (descriptor=="surf") fdetector=cv::xfeatures2d::SURF::create();
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    std::vector<cv::Mat>    features;
    //std::cout << "Extracting   features..." << std::endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        //std::cout<<"reading image: "<<path_to_images[i]<<std::endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        if (bFlip){
            cv::flip(image, image, 1);
        }
        
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        kps_list.push_back(keypoints);
    }
    return features;
}

void testVocCreation(const std::vector<cv::Mat> &features)
{
    const int k = 15;
    const int L = 6;
    const DBoW3::WeightingType weight =  DBoW3::TF_IDF;
    const DBoW3::ScoringType score =  DBoW3::L1_NORM;
    DBoW3::Vocabulary voc(k, L, weight, score);
    voc.create(features);
    std::cout << "Vocabulary information: " << std::endl<< voc << std::endl << std::endl;
    voc.save("small_voc.bin");
}

void show_match(std::vector<cv::DMatch> good_matches, cv::Mat img1, cv::Mat img2, std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2){
    cv::Mat canvas;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, canvas);
    cv::imshow("img1", canvas);
    
    cv::waitKey(-1);
}

std::vector<cv::DMatch> match_desc(cv::Mat descriptors1, cv::Mat descriptors2){
    float dis_thresh=30;
    //cv::BFMatcher matcher(cv::NORM_L2, true);
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    std::sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> good_matches;
    const int ptsPairs = std::min(1000, (int)(matches.size() * 0.2f));
    for( int j = 0; j < ptsPairs; j++ )
    {
        if(matches[j].distance<dis_thresh){
            good_matches.push_back(matches[j]);
        }
    }
    
    std::vector<cv::DMatch> matches1;
    matcher.match(descriptors2, descriptors1, matches1);
    std::sort(matches1.begin(), matches1.end());
    std::vector<cv::DMatch> good_matches1;
    const int ptsPairs1 = std::min(1000, (int)(matches1.size() * 0.2f));
    for( int j = 0; j < ptsPairs1; j++ )
    {
        
        if(matches1[j].distance<dis_thresh){
            good_matches1.push_back(matches1[j]);
        }
    }
    if(debug_mode){
        std::cout<<"min diff: "<<matches1[0].distance<<std::endl;
        std::cout<<"max diff: "<<matches1[matches1.size()-1].distance<<std::endl;
    }
    
    std::vector<cv::DMatch> final_good_matches;
    for(int i=0;i<good_matches.size();i++){
        cv::DMatch match = good_matches[i];
        for(int j=0;j<good_matches1.size();j++){
            cv::DMatch match1 = good_matches1[j];
            if(match1.queryIdx==match.trainIdx && match.queryIdx==match1.trainIdx){
                final_good_matches.push_back(match); 
                break;
            }
        }
    }
    if(debug_mode){
        std::cout<<"match count: "<<final_good_matches.size()<<std::endl;
    }
    
    return final_good_matches;
}

std::vector<std::string> get_file_list(std::string file_list_filename){
    std::ifstream file_file_list_file(file_list_filename);
    std::vector<std::string> file_list_vec;
    while(true){
        std::string str;
        std::getline(file_file_list_file, str);
        if(str==""){
            break;
        }else{
            file_list_vec.push_back(str);
        }
    }
    return file_list_vec;
}

bool copyFile(std::string SRC, std::string DEST)
{
    std::ifstream src(SRC, std::ios::binary);
    std::ofstream dest(DEST, std::ios::binary);
    dest << src.rdbuf();
    return src && dest;
}

int main(int argc,char **argv)
{
    std::string cmd_flag=argv[1];
    std::vector<std::string> file_list_vec = get_file_list(argv[2]);
    int inliner_thres=10;
    int query_count=5;
    if (debug_mode){
        inliner_thres=10;
        query_count=5;
    }
    
    int img_count=10000;
    try{
        if(cmd_flag=="gen"){
            std::vector<std::vector<cv::KeyPoint> > db_kps;
            std::vector<cv::Mat> features = loadFeatures(file_list_vec, "orb", db_kps);  
            testVocCreation(features);
        }else if(cmd_flag=="test"){
            std::vector<std::string> query_list_vec = get_file_list(argv[3]);
            bool bFlip=false;
            std::string sflip= argv[3];
            if (sflip=="true"){
                bFlip=true;
            }
            std::cout<<"start loading bow"<<std::endl;
            DBoW3::Vocabulary voc("small_voc.bin");
            DBoW3::Database db(voc, false, 0);
            std::cout<<"finish loading bow"<<std::endl;
            std::cout<<"start loading db features"<<std::endl;
            std::vector<std::vector<cv::KeyPoint> > db_kps;
            std::vector<cv::Mat> db_features = loadFeatures(file_list_vec, "orb", db_kps);  
            for(size_t i = 0; i < db_features.size(); i++){
                db.add(db_features[i]);
            }
            std::cout<<"finish loading db features"<<std::endl;
            
            std::cout<<"start loading query features"<<std::endl;
            std::vector<std::vector<cv::KeyPoint> > query_kps;
            std::vector<cv::Mat> query_decs = loadFeatures(query_list_vec, "orb", query_kps,bFlip);
            std::cout<<"finish loading query features"<<std::endl;
            for(size_t i = 0; i < query_decs.size(); i++){
                DBoW3::QueryResults ret;
                db.query(query_decs[i], ret, query_count);
                for (size_t re_i=0; re_i<ret.size();re_i++){
                    std::vector<cv::DMatch> good_matches = match_desc(query_decs[i], db_features[ret[re_i].Id]);
                    if (good_matches.size()>=inliner_thres){
                        cv::Mat query_img = cv::imread(query_list_vec[i]);
                        float img_size=sqrt(query_img.cols*query_img.cols+query_img.rows*query_img.rows);
                        if (check_geo_insistence(query_kps[i], db_kps[ret[re_i].Id], good_matches, img_size)){
                            //std::cout<<query_list_vec[i]<<","<<file_list_vec[ret[re_i].Id]<<std::endl;
                            std::cout<<query_list_vec[i]<<std::endl;
                            std::stringstream ss1;
                            ss1<<"./re/"<<img_count<<"_1.jpg";
                            std::stringstream ss2;
                            ss2<<"./re/"<<img_count<<"_2.jpg";
                            copyFile(query_list_vec[i], ss1.str());
                            copyFile(file_list_vec[ret[re_i].Id], ss2.str());
                            remove(query_list_vec[i].c_str());
                            img_count++;
                            break;
                        }
                    }
                    if(debug_mode){
                        std::string match_img_name = file_list_vec[ret[re_i].Id];
                        cv::Mat re_img = cv::imread(match_img_name);
                        cv::Mat query_img = cv::imread(query_list_vec[i]);
                        std::cout << "img: " << match_img_name << ". " << ret[re_i].Score << std::endl;
                        show_match(good_matches, query_img, re_img, query_kps[i], db_kps[ret[re_i].Id]);
                    }
                }
            }
        }
    }catch(std::exception &ex){
        std::cerr<<ex.what()<<std::endl;
    }

    return 0;
}
