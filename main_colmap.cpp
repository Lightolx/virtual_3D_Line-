/*
Line3D++ - Line-based Multi View Stereo
Copyright (C) 2015  Manuel Hofer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// check libs
#include "configLIBS.h"

// EXTERNAL
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
#include <boost/filesystem.hpp>
#include "eigen3/Eigen/Eigen"

// std
#include <iostream>
#include <fstream>

// opencv
#ifdef L3DPP_OPENCV3
#include <opencv2/highgui.hpp>
#else
#include <opencv/highgui.h>
#endif //L3DPP_OPENCV3

// lib
#include "line3D.h"

// INFO:
// This executable reads colmap results (cameras.txt, images.txt, and points3D.txt) and executes the Line3D++ algorithm.
// If distortion coefficients are stored in the cameras.txt file, you need to use the _original_ (distorted) images!

int main(int argc, char *argv[])
{
    using std::cout;
    using std::endl;

    TCLAP::CmdLine cmd("LINE3D++");

    TCLAP::ValueArg<std::string> inputArg("i", "input_folder", "folder containing the images", true, "", "string");
    cmd.add(inputArg);

    TCLAP::ValueArg<std::string> sfmArg("m", "sfm_folder", "full path to the colmap result files (cameras.txt, images.txt, and points3D.txt), if not specified --> input_folder", false, "", "string");
    cmd.add(sfmArg);

    TCLAP::ValueArg<std::string> outputArg("o", "output_folder", "folder where result and temporary files are stored (if not specified --> sfm_folder+'/Line3D++/')", false, "", "string");
    cmd.add(outputArg);

    TCLAP::ValueArg<int> scaleArg("w", "max_image_width", "scale image down to fixed max width for line segment detection", false, L3D_DEF_MAX_IMG_WIDTH, "int");
    cmd.add(scaleArg);

    TCLAP::ValueArg<int> neighborArg("n", "num_matching_neighbors", "number of neighbors for matching", false, L3D_DEF_MATCHING_NEIGHBORS, "int");
    cmd.add(neighborArg);

    TCLAP::ValueArg<float> sigma_A_Arg("a", "sigma_a", "angle regularizer", false, L3D_DEF_SCORING_ANG_REGULARIZER, "float");
    cmd.add(sigma_A_Arg);

    TCLAP::ValueArg<float> sigma_P_Arg("p", "sigma_p", "position regularizer (if negative: fixed sigma_p in world-coordinates)", false, L3D_DEF_SCORING_POS_REGULARIZER, "float");
    cmd.add(sigma_P_Arg);

    TCLAP::ValueArg<int>  knnArg("k", "knn_matches", "number of matches to be kept (<= 0  --> use all that fulfill overlap)", false, L3D_DEF_KNN, "int");
    cmd.add(knnArg);

    TCLAP::ValueArg<int> segNumArg("y", "num_segments_per_image", "maximum number of 2D segments per image (longest)", false, L3D_DEF_MAX_NUM_SEGMENTS, "int");
    cmd.add(segNumArg);

    TCLAP::ValueArg<int> visibilityArg("v", "visibility_t", "minimum number of cameras to see a valid 3D line", false, L3D_DEF_MIN_VISIBILITY_T, "int");
    cmd.add(visibilityArg);

    TCLAP::ValueArg<bool> diffusionArg("d", "diffusion", "perform Replicator Dynamics Diffusion before clustering", false, L3D_DEF_PERFORM_RDD, "bool");
    cmd.add(diffusionArg);

    TCLAP::ValueArg<bool> loadArg("l", "load_and_store_flag", "load/store segments (recommended for big images)", false, L3D_DEF_LOAD_AND_STORE_SEGMENTS, "bool");
    cmd.add(loadArg);

    TCLAP::ValueArg<float> collinArg("r", "collinearity_t", "threshold for collinearity", false, L3D_DEF_COLLINEARITY_T, "float");
    cmd.add(collinArg);

    TCLAP::ValueArg<bool> cudaArg("g", "use_cuda", "use the GPU (CUDA)", false, true, "bool");
    cmd.add(cudaArg);

    TCLAP::ValueArg<bool> g2oArg("c", "use_g2o", "use g2o (for 3D line optimization)", false, L3D_DEF_USE_G2O, "bool");
    cmd.add(g2oArg);

    TCLAP::ValueArg<float> constRegDepthArg("z", "const_reg_depth", "use a constant regularization depth (only when sigma_p is metric!)", false, -1.0f, "float");
    cmd.add(constRegDepthArg);

    // read arguments
    cmd.parse(argc,argv);
    std::string inputFolder = inputArg.getValue().c_str();
    std::string sfmFolder = sfmArg.getValue().c_str();

    if(sfmFolder.length() == 0)
        sfmFolder = inputFolder;

    // check if colmap result folder
    boost::filesystem::path sfm(sfmFolder);
    if(!boost::filesystem::exists(sfm))
    {
        std::cerr << "colmap result folder " << sfm << " does not exist!" << std::endl;
        return -1;
    }

    std::string outputFolder = outputArg.getValue().c_str();
    if(outputFolder.length() == 0)
        outputFolder = sfmFolder+"/Line3D++/";

//    int maxWidth = scaleArg.getValue();
    unsigned int neighbors = std::max(neighborArg.getValue(),2);
    bool diffusion = diffusionArg.getValue();
    bool loadAndStore = loadArg.getValue();
    float collinearity = collinArg.getValue();
    bool useGPU = cudaArg.getValue();
    bool useG2O = g2oArg.getValue();
    float sigmaA = fabs(sigma_A_Arg.getValue());
    float sigmaP = sigma_P_Arg.getValue();
    int kNN = knnArg.getValue();
    unsigned int maxNumSegments = segNumArg.getValue();
    unsigned int visibility_t = visibilityArg.getValue();
    float constRegDepth = constRegDepthArg.getValue();

    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);

    // create Line3D++ object
    L3DPP::Line3D* Line3D = new L3DPP::Line3D(outputFolder,loadAndStore,maxNumSegments,true,useGPU);

    // check if result files exist
    boost::filesystem::path sfm_cameras(sfmFolder+"/cameras.txt");
    boost::filesystem::path sfm_images(sfmFolder+"/images.txt");
//    boost::filesystem::path sfm_points3D(sfmFolder+"/points3D.txt");
//    boost::filesystem::path sfm_vps(sfmFolder+"/vp.txt");
    if(!boost::filesystem::exists(sfm_cameras) || !boost::filesystem::exists(sfm_images))
    {
        std::cerr << "at least one of the colmap result files does not exist in sfm folder: " << sfm << std::endl;
        return -2;
    }

    std::cout << std::endl << "reading colmap result..." << std::endl;

    // read cameras.txt
    std::ifstream cameras_file;
    cameras_file.open(sfm_cameras.c_str());
    std::string cameras_line;

    std::map<unsigned int,Eigen::Matrix3d> cams_K;  // store the intrinsic of each camera, [cameraID, cameraK]
    std::map<unsigned int,Eigen::Vector3d> cams_radial;  // store the distortion parameter of each camera
    std::map<unsigned int,Eigen::Vector2d> cams_tangential;

    while(std::getline(cameras_file,cameras_line))
    {
        // check first character for a comment (#)
        if(cameras_line.substr(0,1).compare("#") != 0)
        {
            std::stringstream cameras_stream(cameras_line);

            unsigned int camID,width,height;
            std::string model;

            // parse essential data
            cameras_stream >> camID >> model >> width >> height;

            double fx,fy,cx,cy,k1,k2,k3,p1,p2;

            // check camera model
            if(model.compare("SIMPLE_PINHOLE") == 0)
            {
                // f,cx,cy
                cameras_stream >> fx >> cx >> cy;
                fy = fx;
                k1 = 0; k2 = 0; k3 = 0;
                p1 = 0; p2 = 0;
            }
            else if(model.compare("PINHOLE") == 0)
            {
                // fx,fy,cx,cy
                cameras_stream >> fx >> fy >> cx >> cy;
                k1 = 0; k2 = 0; k3 = 0;
                p1 = 0; p2 = 0;
            }
            else if(model.compare("SIMPLE_RADIAL") == 0)
            {
                // f,cx,cy,k
                cameras_stream >> fx >> cx >> cy >> k1;
                fy = fx;
                k2 = 0; k3 = 0;
                p1 = 0; p2 = 0;
            }
            else if(model.compare("RADIAL") == 0)
            {
                // f,cx,cy,k1,k2
                cameras_stream >> fx >> cx >> cy >> k1 >> k2;
                fy = fx;
                k3 = 0;
                p1 = 0; p2 = 0;
            }
            else if(model.compare("OPENCV") == 0)
            {
                // fx,fy,cx,cy,k1,k2,p1,p2
                cameras_stream >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2;
                k3 = 0;
            }
            else if(model.compare("FULL_OPENCV") == 0)
            {
                // fx,fy,cx,cy,k1,k2,p1,p2,k3[,k4,k5,k6]
                cameras_stream >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2 >> k3;
            }
            else
            {
                std::cerr << "camera model " << model << " unknown!" << std::endl;
                std::cerr << "please specify its parameters in the main_colmap.cpp in order to proceed..." << std::endl;
                return -3;
            }

            Eigen::Matrix3d K;
            K(0,0) = fx; K(0,1) = 0;  K(0,2) = cx;
            K(1,0) = 0;  K(1,1) = fy; K(1,2) = cy;
            K(2,0) = 0;  K(2,1) = 0;  K(2,2) = 1;

            cams_K[camID] = K;
            cams_radial[camID] = Eigen::Vector3d(k1,k2,k3);
            cams_tangential[camID] = Eigen::Vector2d(p1,p2);
        }
    }
    cameras_file.close();

    std::cout << "found " << cams_K.size() << " cameras in [cameras.txt]" << std::endl;

    // read images.txt
    std::ifstream images_file;
    images_file.open(sfm_images.c_str());
    std::string images_line;

    std::map<unsigned int,Eigen::Matrix3d> img_R;  // store the pose of each image, [imageID, imageR]
    std::map<unsigned int,Eigen::Vector3d> img_t;  // [imageID, imaget]
    std::map<unsigned int,Eigen::Vector3d> img_C;  // [imageID, imageC]
    std::map<unsigned int,unsigned int> img2cam;    // [imageID, cameraID], i.e., the camera shoots this image
    std::map<unsigned int,std::string> imgID_imgName; // [imageID, image name]
//    std::map<unsigned int,std::list<unsigned int> > cams_worldpoints; // world point coordinates of each image, [imageID, pointIDs],
                                                                      // like [1, (3, 9 ,17, ...)]
//    std::map<unsigned int,Eigen::Vector3d> wps_coords; // world points coordinate, [world point ID, world point coordinate]
    std::vector<unsigned int> img_seq;              // image sequence, generated by the image.txt, like (1, 2, 3, ...)
    unsigned int imgID,camID;

    while(std::getline(images_file,images_line))
    {
        // check first character for a comment (#)
        if(images_line.substr(0,1).compare("#") != 0)
        {
            std::stringstream images_stream(images_line);

            // image data
            double qw,qx,qy,qz,tx,ty,tz;
            std::string img_name;

            images_stream >> imgID >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camID >> img_name;

            // convert rotation
            if(cams_K.find(camID) != cams_K.end())
            {
                Eigen::Matrix3d R = Line3D->rotationFromQ(qw,qx,qy,qz);
                Eigen::Vector3d t(tx,ty,tz);
                Eigen::Vector3d C = (R.transpose()) * (-1.0 * t);

                img_R[imgID] = R;
                img_t[imgID] = t;
                img_C[imgID] = C;
                imgID_imgName[imgID] = img_name;
                img2cam[imgID] = camID;
                img_seq.push_back(imgID);
            }
        }
    }
    images_file.close();

    std::cout << "found " << img_R.size() << " images in [images.txt]" << std::endl;

    //calculate the vanish point
    std::map<unsigned int,Eigen::Vector2d> vps;
    vps = Line3D->calVanishPoint(img_R, img_t, cams_K[1]);


    /*
    // read points3D.txt
    std::ifstream points3D_file;
    points3D_file.open(sfm_points3D.c_str());
    std::string points3D_line;

    while(std::getline(points3D_file,points3D_line))
    {
        // check first character for a comment (#)
        if(images_line.substr(0,1).compare("#") != 0)
        {
            std::stringstream points3D_stream(points3D_line);

            // read id and coords
            double X,Y,Z;
            unsigned int pID;

            points3D_stream >> pID >> X >> Y >> Z;

            if(wps_coords.find(pID) != wps_coords.end())
            {
                wps_coords[pID] = Eigen::Vector3d(X,Y,Z);
            }
        }
    }
    points3D_file.close();
    */

    /*
    // read vps.txt
    std::ifstream vps_file;
    vps_file.open(sfm_vps.c_str());
    std::string vps_line;

    std::map<unsigned int,Eigen::Vector2d> img_vp;  // [imageID, vanish point coordinate]

    while(std::getline(vps_file, vps_line))
    {
        // check first character for a comment (#)
        if(cameras_line.substr(0,1).compare("#") != 0)
        {
            Eigen::Vector2d vp;
            int imgID = -1;

            std::stringstream ss(vps_line);
            ss >> imgID >> vp(0) >> vp(1);

            if (imgID != -1)
            {
                img_vp[imgID] = vp*2;
            }
        }
    }

    vps_file.close();
     */

    // load images (parallel)
    inputFolder += "/images";
#ifdef L3DPP_OPENMP
    #pragma omp parallel for
#endif //L3DPP_OPENMP
    for(unsigned int i=0; i<img_seq.size(); ++i)
    {
        // get camera params
        unsigned int imgID = img_seq[i];
        unsigned int camID = img2cam[imgID];

        // intrinsics
        Eigen::Matrix3d K = cams_K[camID];
        Eigen::Vector3d radial = cams_radial[camID];
        Eigen::Vector2d tangential = cams_tangential[camID];

        if(img_R.find(imgID) != img_R.end())
        {
            // extrinsics
            Eigen::Matrix3d R = img_R[imgID];
            Eigen::Vector3d t = img_t[imgID];
            Eigen::Vector3d C = img_C[imgID];

            // read image
//            cout << "inputFolder is " << inputFolder << ", cams_images[ID] is " << cams_images[imgID] << endl;

            cv::Mat image = cv::imread(inputFolder+"/"+imgID_imgName[imgID],CV_LOAD_IMAGE_COLOR);

            // undistort image
            cv::Mat img_undist;
            if(fabs(radial(0)) > L3D_EPS || fabs(radial(1)) > L3D_EPS || fabs(radial(2)) > L3D_EPS ||
                    fabs(tangential(0)) > L3D_EPS || fabs(tangential(1)) > L3D_EPS)
            {
                // undistorting
                Line3D->undistortImage(image,img_undist,radial,tangential,K);
            }
            else
            {
                // already undistorted
                img_undist = image;
            }

            // add image
            Line3D->addImage(imgID,img_undist,K,R,t,vps[imgID]);

            /*
            // compute depths
            if(cams_worldpoints.find(imgID) != cams_worldpoints.end())
            {
                std::list<unsigned int> wps_list = cams_worldpoints[imgID];
                std::vector<float> depths;

                std::list<unsigned int>::iterator it = wps_list.begin();
                for(; it!=wps_list.end(); ++it)
                {
                    depths.push_back((C-wps_coords[*it]).norm());
                }

                // median depth of all points in this image
                if(depths.size() > 0)
                {
                    std::sort(depths.begin(),depths.end());
                    float med_depth = depths[depths.size()/2];


                }
            }
             */
        }
    }

    // match images
    Line3D->matchImages(sigmaP,sigmaA,neighbors,kNN,constRegDepth);

    // compute result
    Line3D->reconstruct3Dlines(visibility_t,diffusion,collinearity,useG2O);

    // save end result
    std::vector<L3DPP::FinalLine3D> result;
    Line3D->get3Dlines(result);

    // save as STL
    Line3D->saveResultAsSTL(outputFolder);
    // save as OBJ
    Line3D->saveResultAsOBJ(outputFolder);
    // save as TXT
    Line3D->save3DLinesAsTXT(outputFolder);
    // save as BIN
    Line3D->save3DLinesAsBIN(outputFolder);

    // cleanup
    delete Line3D;
}
