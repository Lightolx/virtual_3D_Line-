#include <cv.hpp>
#include "line3D.h"

#include <sstream>

namespace L3DPP
{
    //------------------------------------------------------------------------------
    Line3D::Line3D(const std::string& output_folder, const bool load_segments,
                   const unsigned int max_line_segments,
                   const bool neighbors_by_worldpoints,
                   const bool use_GPU) :
        data_folder_(output_folder+"L3D++_data/"), load_segments_(load_segments),
        max_line_segments_(max_line_segments), neighbors_by_worldpoints_(neighbors_by_worldpoints)

    {
        // set params
        num_lines_total_ = 0;
        med_scene_depth_ = L3D_EPS;
        med_scene_depth_lines_ = 0.0f;
        translation_ = Eigen::Vector3d(0,0,0);

        // default
        collinearity_t_ = L3D_DEF_COLLINEARITY_T;
        num_neighbors_ = L3D_DEF_MATCHING_NEIGHBORS;
        epipolar_overlap_ = L3D_DEF_EPIPOLAR_OVERLAP;
        kNN_ = L3D_DEF_KNN;
        sigma_p_ = L3D_DEF_SCORING_POS_REGULARIZER;
        sigma_a_ = L3D_DEF_SCORING_ANG_REGULARIZER;
        const_regularization_depth_ = -1.0f;
        two_sigA_sqr_ = 2.0f*sigma_a_*sigma_a_;
        perform_RDD_ = false;
        use_G2O_ = false;
        max_iter_G2O_ = L3D_DEF_G2O_MAX_ITER;
        visibility_t_ = 3;

        if(sigma_p_ < L3D_EPS)
        {
            // fixed sigma_p in world-coords
            fixed3Dregularizer_ = true;
            sigma_p_ = fabs(sigma_p_);
        }
        else
        {
            // regularizer in pixels (scale unknown)
            fixed3Dregularizer_ = false;
            sigma_p_ = fmax(0.1f,sigma_p_);
        }

#ifdef L3DPP_CUDA
        useGPU_ = use_GPU;
#else
        useGPU_ = false;
#endif //L3DPP_CUDA

        prefix_ = "[L3D++] ";
        prefix_err_ = prefix_+"ERROR: ";
        prefix_wng_ = prefix_+"WARNING: ";

        // create output directory
        boost::filesystem::path dir(data_folder_);
        boost::filesystem::create_directory(dir);

        std::cout << std::endl;
        std::cout << prefix_ << "//////////////////////////////////////////////////////////////////////" << std::endl;
        std::cout << prefix_ << "Line3D++ - http://www.icg.tugraz.at/ - AerialVisionGroup" << std::endl;
        std::cout << prefix_ << "(c) 2015, Manuel Hofer" << std::endl;
        std::cout << prefix_ << "published under the GNU General Public License" << std::endl;
        std::cout << prefix_ << "//////////////////////////////////////////////////////////////////////" << std::endl;
    }

    //------------------------------------------------------------------------------
    Line3D::~Line3D()
    {
        // delete views
        std::map<unsigned int,L3DPP::View*>::iterator it = views_.begin();
        for(; it!=views_.end(); ++it)
        {
            delete it->second;
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::undistortImage(const cv::Mat& inImg, cv::Mat& outImg,
                                const Eigen::Vector3d& radial_coeffs,
                                const Eigen::Vector2d& tangential_coeffs,
                                const Eigen::Matrix3d& K)
    {
        cv::Mat I = cv::Mat_<double>::eye(3,3);
        cv::Mat cvK = cv::Mat_<double>::zeros(3,3);
        cvK.at<double>(0,0) = K(0,0);
        cvK.at<double>(1,1) = K(1,1);
        cvK.at<double>(0,2) = K(0,2);
        cvK.at<double>(1,2) = K(1,2);
        cvK.at<double>(2,2) = 1.0;

        cv::Mat cvDistCoeffs(5,1,CV_64FC1,cv::Scalar(0));
        cvDistCoeffs.at<double>(0) = radial_coeffs.x();
        cvDistCoeffs.at<double>(1) = radial_coeffs.y();
        cvDistCoeffs.at<double>(2) = tangential_coeffs.x();
        cvDistCoeffs.at<double>(3) = tangential_coeffs.y();
        cvDistCoeffs.at<double>(4) = radial_coeffs.z();

        cv::Mat undistort_map_x;
        cv::Mat undistort_map_y;

        cv::initUndistortRectifyMap(cvK,cvDistCoeffs,I,cvK,cv::Size(inImg.cols, inImg.rows),
                                    undistort_map_x.type(), undistort_map_x, undistort_map_y );
        cv::remap(inImg,outImg,undistort_map_x,undistort_map_y,cv::INTER_LINEAR,cv::BORDER_CONSTANT);
    }

    //------------------------------------------------------------------------------
    void Line3D::addImage(const unsigned int imgID, cv::Mat& image,
                          const Eigen::Matrix3d& K, const Eigen::Matrix3d& R,
                          const Eigen::Vector3d& t, const Eigen::Vector2d vp,
                          const std::vector<cv::Vec4f>& line_segments)
    {
        // check size
//        std::cout << "image.cols is " << image.cols << ", image.rows is " << image.rows << std::endl;
        if(std::max(image.cols,image.rows) < L3D_DEF_MIN_IMG_WIDTH)
        {
            display_text_mutex_.lock();
            std::cout << prefix_err_ << "image is too small for reliable results: " << std::max(image.cols,image.rows);
            std::cout << "px (larger side should be >= " << L3D_DEF_MIN_IMG_WIDTH << "px)" << std::endl;
            display_text_mutex_.unlock();
            return;
        }

        // check ID
        view_reserve_mutex_.lock();
        if(views_reserved_.find(imgID) != views_reserved_.end())
        {
            display_text_mutex_.lock();
            std::cout << prefix_err_ << "image ID [" << imgID << "] already in use!" << std::endl;
            display_text_mutex_.unlock();

            view_reserve_mutex_.unlock();
            return;
        }
        else
        {
            // reserve
            views_reserved_.insert(imgID);
        }

        if(views_reserved_.size() == 1)
        {
            display_text_mutex_.lock();
            std::cout << std::endl << prefix_ << "[1] ADDING IMAGES ================================" << std::endl;
            display_text_mutex_.unlock();
        }
        view_reserve_mutex_.unlock();

        /*
        // check worldpoints
        if(wps_or_neighbors.size() == 0)
        {
            display_text_mutex_.lock();
            if(neighbors_by_worldpoints_)
                std::cout << prefix_err_ << "view [" << imgID << "] has no worldpoints!" << std::endl;
            else
                std::cout << prefix_err_ << "view [" << imgID << "] has no visual neighbors!" << std::endl;

            display_text_mutex_.unlock();

            return;
        }
         */

        /*
        // output optical center
        std::ofstream fout;
        fout.open("/media/psf/Home/Desktop/center.txt", std::ios::app);
        Eigen::Vector3d C = -R.inverse()*t;
        fout << C.x() << " " << C.y() << " " << C.z() << std::endl;
        fout.close();
         */

//        Eigen::Vector3d C = -R.inverse()*t;
//        std::ofstream fout("/media/psf/Home/Desktop/center.txt", std::ios::app);
//        fout << C.x() << " " << C.y() << " " << C.z() << endl;
//        fout.close();

        // detect segments
        L3DPP::DataArray<float4>* lines = NULL;  // line segments of this image, include the coordinate of p1,p2 in dataCPU
        // detect segments using LSD algorithm
//            lines = detectLineSegments(imgID,image);
        // add visual lines
        int h = image.rows;
        int w = image.cols;
        double ka = vp(0)/(vp(1) - h);
        double ta = -double(h)*ka;
        double kb = (vp(0)-w)/(vp(1)-h);
        double tb = -double(h)*kb + w;

        int height_up = 550;
        int height_down = 700;
        cv::Point2f srcQuad[4];  // set four corner point empirically, height is 550 and 700, respectively
        srcQuad[0].y = height_up;srcQuad[0].x = ka*srcQuad[0].y + ta;
        srcQuad[1].y = height_down;srcQuad[1].x = ka*srcQuad[1].y + ta;
        srcQuad[2].y = height_up;srcQuad[2].x = kb*srcQuad[2].y + tb;
        srcQuad[3].y = height_down;srcQuad[3].x = kb*srcQuad[3].y + tb;
        // todo:: watch out! when the line num increase, layering may occur
        int line_num = 10;
        lines = addVirtualLines(image, srcQuad, line_num);

        // find ROI
        // todo:: how to iterate h??? it is the most important things now
        double cam_height = 1.5; // set camera height to be 2.0m
        Eigen::Vector3d vpX = K.inverse()*Eigen::Vector3d(vp(0), vp(1), 1);  // vanish point on normlized plane
        Eigen::Vector3d CX = Eigen::Vector3d(0, 0, 1); // optical center on normalized plane
        double pitch = acos(vpX.dot(CX)/(vpX.squaredNorm()*CX.squaredNorm()));
        Eigen::Vector3d nVec(0, cos(pitch), sin(pitch));  // road plane norm vector in camera frame
        Eigen::Vector3d ypr = (R.inverse()).eulerAngles(2,1,0);
        Eigen::Vector3d rpy = R.eulerAngles(1,2,0);
        Eigen::Vector3d yyy = (R.inverse()).eulerAngles(1,2,0);
        Eigen::Vector3d zzz = (R.inverse()).eulerAngles(0,2,1);
        double roll = 3.1415926-yyy[0];
        double pitch1 = 3.1415926-yyy[1];
        double yaw = 3.1415926+yyy[2];
        yyy[0] = 3.1415926-yyy[0];
        yyy[1] = 3.1415926-yyy[1];
        yyy[2] = 3.1415926+yyy[2];
        Eigen::Matrix3d Rnew = this->rotationFromRPY(ypr.z(), ypr.y(), ypr.x());
//        const Eigen::AngleAxisd Rx
//                = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
//        const Eigen::AngleAxisd Ry
//                = Eigen::AngleAxisd(pitch1, Eigen::Vector3d::UnitY());
//        const Eigen::AngleAxisd Rz
//                = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
//        std::cout << "original R is " << (Rx*Ry*Rz).matrix() << endl;

//        rpy[2] = pitch;
//        yyy[2] += 3.1415926;
//        yyy[1] = 3.1415926-yyy[1];
//        yyy[0] = 0;
        Eigen::Matrix3d Rnew2 = this->rotationFromRPY(yyy);
//        Eigen::Matrix3d Rnew2 = this->rotationFromPRY(pry.z(),pry.y(),pry.x());
        std::cout << "R is\n" << R << "\nRnew is\n" << Rnew << "\nRnew2 is\n" << Rnew2 << std::endl;
//        std::cout << "R is\n" << R;
        // find ROI
        Eigen::Vector3d srcQ[4];
        Eigen::Vector3d Pw[4];       // corresponding rectangle region of these lines in world coordinate
        for (int i = 0; i < 4; ++i)  // homogeneous
        {
            srcQ[i](0) = srcQuad[i].x;
            srcQ[i](1) = srcQuad[i].y;
            srcQ[i](2) = 1;
            Eigen::Vector3d x2d = srcQ[i];              // point on image plane
            Eigen::Vector3d x3dc = K.inverse()*x2d;     // arbitrary 3d point in camera frame
            volatile double Yproject = nVec.dot(x3dc);
            double lamda = cam_height/(nVec.dot(x3dc));     // scale between x3dc and the point on road plane
            Eigen::Vector3d x3dcp = lamda*x3dc;// 3d point on road plane in camera frame
            volatile double Yproject2 = x3dcp.dot(nVec);
            Eigen::Vector3d point3dw = Rnew2.inverse()*(x3dcp - t);  // 3d point in world coordinate
            Pw[i] = point3dw;
            Eigen::Vector3d C = -R.inverse()*t;
//            double altitude_diff = C.y() - point3dw.y();
            Eigen::Vector3d diff = Rnew2.inverse()*x3dcp;
            Eigen::Vector3d diff1 = Rnew2*x3dcp;
            double altitude_diff = (Rnew2.inverse()*x3dcp).y();
            if (i == 1)
            cout << "altitude_diff is " << altitude_diff << endl;
        }

        // assign each line the 3D coordinate of its center
        Eigen::Vector3d central(0,0,0);
        Eigen::Vector3d centrals[line_num];  // centers of each virtual line
        for (int i = 0; i < line_num; ++i)
        {
            central.x() = (lines->dataCPU(i,0)[0].x + lines->dataCPU(i,0)[0].z)/2;
            central.y() = (lines->dataCPU(i,0)[0].y + lines->dataCPU(i,0)[0].w)/2;
            central.z() = 1;
            Eigen::Vector3d x2d = central;              // point on image plane
            Eigen::Vector3d x3dc = K.inverse()*x2d;     // arbitrary 3d point in camera frame
            double lamda = cam_height/(nVec.dot(x3dc));     // scale between x3dc and the point on road plane
            Eigen::Vector3d x3dcp = lamda*x3dc;         // 3d point on road plane in camera frame
            Eigen::Vector3d point3dw = R.inverse()*(x3dcp - t);  // 3d point in world coordinate
            centrals[i] = point3dw;
        }

        /*
        // show image and the virtual line
        for (int i = 0; i < lines->width(); ++i)
        {
            float4 coordsf4 = lines->dataCPU(i,0)[0];
//            cout << coordsf4.x << ", " << coordsf4.y << ", " << coordsf4.w << ", " << coordsf4.z << endl;
            cv::Point2f a(coordsf4.x, coordsf4.y);
            cv::Point2f b(coordsf4.z, coordsf4.w);
            cv::circle(image, cv::Point2d(vp(0), vp(1)), 8, cv::Scalar(0,255,0));
//            cv::putText(image,std::to_string(i), a, 1, 2, cv::Scalar(0, 255, 255));
            cv::line(image, a, b, cv::Scalar(0, 255, 255), 2);
        }

        cv::imshow("virtual line", image);
        cv::waitKey(1);
         */

        if(lines == NULL)
        {
            display_text_mutex_.lock();
            std::cout << prefix_wng_ << "no line segments found in image [" << imgID << "]!" << std::endl;
            display_text_mutex_.unlock();

            return;
        }

        // create view for this image
        L3DPP::View* v = new L3DPP::View(imgID,lines,K,R,t,image.cols,image.rows,Pw,centrals,line_num);
        view_mutex_.lock();

        display_text_mutex_.lock();
//        std::cout << prefix_ << "adding view [" << std::setfill('0') << std::setw(L3D_DISP_CAMS) << camID;
//        std::cout << "]: #lines = " << std::setfill(' ') << std::setw(L3D_DISP_LINES) << lines->width();
//        std::cout << " [" << std::setfill('0') << std::setw(L3D_DISP_CAMS) << views_.size() << "]" << std::endl;
        display_text_mutex_.unlock();

        views_[imgID] = v;
        view_order_.push_back(imgID);
        matches_[imgID] = std::vector<std::list<L3DPP::Match> >(lines->width());
        num_matches_[imgID] = 0;
        processed_[imgID] = false;
        visual_neighbors_[imgID] = std::set<unsigned int>();
        num_lines_total_ += lines->width();

        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::processWPlist(const unsigned int imgID, const std::list<unsigned int>& wps)
    {
        std::list<unsigned int>::const_iterator it = wps.begin();
        for(; it!=wps.end(); ++it)
        {
            unsigned int wpID = *it;
            worldpoints2views_[wpID].push_back(imgID);
        }
        num_worldpoints_[imgID] = wps.size();
        views2worldpoints_[imgID] = wps;
    }

    //------------------------------------------------------------------------------
    void Line3D::setVisualNeighbors(const unsigned int camID, const std::list<unsigned int>& neighbors)
    {
        fixed_visual_neighbors_[camID] = neighbors;
    }

    //------------------------------------------------------------------------------
    L3DPP::DataArray<float4>* Line3D::detectLineSegments(const unsigned int camID, const cv::Mat& image)
    {
        /*
        // check image format
        cv::Mat imgGray;
        if(image.type() == CV_8UC3)
        {
            // convert to grayscale
            cv::cvtColor(image,imgGray,CV_RGB2GRAY);
        }
        else if(image.type() == CV_8U)
        {
            imgGray = image.clone();
        }
        else
        {
            display_text_mutex_.lock();
            std::cout << prefix_err_ << "image type not supported! must be CV_8U (gray) or CV_8UC3 (RGB)!" << std::endl;
            display_text_mutex_.unlock();
            return NULL;
        }

        // check image size
        int max_dim = std::max(imgGray.rows,imgGray.cols);
        float upscale_x = 1.0f;
        float upscale_y = 1.0f;
        unsigned int new_width = imgGray.cols;
        unsigned int new_height = imgGray.rows;

        cv::Mat imgResized;
        if(max_image_width_ > 0 && max_dim > max_image_width_)
        {
            // rescale
            float s = float(max_image_width_)/float(max_dim);
            cv::resize(imgGray,imgResized,cv::Size(),s,s);

            upscale_x = float(imgGray.cols)/float(imgResized.cols);
            upscale_y = float(imgGray.rows)/float(imgResized.rows);

            new_width = imgResized.cols;
            new_height = imgResized.rows;
        }
        else
        {
            imgResized = imgGray.clone();
        }

        // see if lines already exist
        L3DPP::DataArray<float4>* segments = NULL;  // line segment, include the coordinate of p1,p2 in dataCPU
        std::stringstream str;
        if(load_segments_)
        {
            str << data_folder_ << "segments_L3D++_/" << camID << "_" << new_width << "x" << new_height << "_" << L3D_DEF_MAX_NUM_SEGMENTS << ".bin";

            boost::filesystem::path file(str.str());
            if(boost::filesystem::exists(file))
            {
                segments = new L3DPP::DataArray<float4>();
                L3DPP::serializeFromFile(str.str(),*segments);
                return segments;
            }
        }

        // detect line segments
#ifndef L3DPP_OPENCV3
        cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetectorPtr(cv::LSD_REFINE_ADV);
#else
        cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
#endif //L3DPP_LSD_EXT
        std::vector<cv::Vec4f> detections;
        lsd->detect(imgResized,detections);

        float diag = sqrtf(float(image.rows*image.rows)+float(image.cols*image.cols));
        float min_len = diag*L3D_DEF_MIN_LINE_LENGTH_FACTOR;

        cv::Mat tmpImage = image.clone();

        L3DPP::lines2D_sorted_by_length sorted;
        for(size_t i=0; i<detections.size(); ++i)
        {
            cv::Vec4f data = detections[i];

            L3DPP::SegmentData2D seg2D;
            seg2D.p1x_ = data(0)*upscale_x;
            seg2D.p1y_ = data(1)*upscale_y;
            seg2D.p2x_ = data(2)*upscale_x;
            seg2D.p2y_ = data(3)*upscale_y;

            /*
            int p1x, p1y, p2x, p2y;
            p1x = round(data(0)*upscale_x);
            p1y = round(data(1)*upscale_y);
            p2x = round(data(2)*upscale_x);
            p2y = round(data(3)*upscale_y);

            int X;
            int Y;
            float intervalX = (p2x - p1x) / float(100);
            float intervalY = (p2y - p1y) / float(100);
            for (int j = 0; j < 101; ++j)
            {
                X = p1x + round(j * intervalX);
                Y = p1y + round(j * intervalY);
                tmpImage.at<uchar>(Y, X) = 0;
            }


            float dx = seg2D.p1x_-seg2D.p2x_;
            float dy = seg2D.p1y_-seg2D.p2y_;
            seg2D.length_ = sqrtf(dx*dx + dy*dy);

            if(seg2D.length_ > min_len)
                sorted.push(seg2D);
        }

//        std::stringstream s;
//        s << camID;
//        cv::imshow(s.str().c_str(), tmpImage);
//        cv::waitKey();

        if(sorted.size() > 0)
        {
            // convert to dataArray
            if(sorted.size() < max_line_segments_)
                segments = new L3DPP::DataArray<float4>(sorted.size(),1);
            else
                segments = new L3DPP::DataArray<float4>(max_line_segments_,1);

//            volatile int tmp = sorted.size();
            unsigned int pos = 0;
            while(!sorted.empty() && pos < segments->width())
            {
                L3DPP::SegmentData2D segData = sorted.top();
                float4 coordsf4;
                coordsf4.x = segData.p1x_; coordsf4.y = segData.p1y_;
                coordsf4.z = segData.p2x_; coordsf4.w = segData.p2y_;
                segments->dataCPU(pos,0)[0] = coordsf4;
                sorted.pop();
                ++pos;
            }

            // save
//            if(load_segments_)
//            {
//                std::cout << str.str() << std::endl;
//                L3DPP::serializeToFile(str.str(),*segments);
//            }

            return segments;
        }

        return NULL;
         */
    }

    //------------------------------------------------------------------------------
    void Line3D::matchImages(const float sigma_position, const float sigma_angle,
                             const unsigned int num_neighbors, const int kNN,
                             const float const_regularization_depth)
    {
        // no new views can be added in the meantime!
        view_reserve_mutex_.lock();
        view_mutex_.lock();

        std::cout << std::endl << prefix_ << "[2] LINE MATCHING ================================" << std::endl;

        if(views_.size() == 0)
        {
            std::cout << prefix_wng_ << "no images to match! forgot to add them?" << std::endl;
            view_mutex_.unlock();
            view_reserve_mutex_.unlock();
            return;
        }

        // check params
        num_neighbors_ = std::max(int(num_neighbors),2);
        sigma_p_ = sigma_position;
        sigma_a_ = fmin(fabs(sigma_angle),90.0f);
        two_sigA_sqr_ = 2.0f*sigma_a_*sigma_a_;
        epipolar_overlap_ = fmin(fabs(1),0.99f);
        kNN_ = kNN;
        const_regularization_depth_ = const_regularization_depth;

        if(sigma_p_ < 0.0f)
        {
            // fixed sigma_p in world-coords
            fixed3Dregularizer_ = true;
            sigma_p_ = fabs(sigma_p_);
        }
        else
        {
            // regularizer in pixels (scale unknown)
            fixed3Dregularizer_ = false;
            sigma_p_ = fmax(0.1f,sigma_p_);
        }

        // reset
        matched_.clear();
        estimated_position3D_.clear();
        entry_map_.clear();

        // compute spatial regularizer
        if(!fixed3Dregularizer_)
            std::cout << prefix_ << "computing spatial regularizers... [" << sigma_p_ << " px]" << std::endl;
        else
            std::cout << prefix_ << "computing spatial regularizers... [" << sigma_p_ << " m]" << std::endl;

        med_scene_depth_ = const_regularization_depth_;
        if(const_regularization_depth_ < 0.0f && fixed3Dregularizer_ && views_avg_depths_.size() > 0)
        {
            // compute median scene depth
            std::sort(views_avg_depths_.begin(),views_avg_depths_.end());
            med_scene_depth_ = views_avg_depths_[views_avg_depths_.size()/2];
            std::cout << prefix_ << "median_scene_depth = " << med_scene_depth_ << std::endl;
        }

        // translate reconstruction (for better numerical stability)
        translate();

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<view_order_.size(); ++i)
        {
            unsigned int imgID = view_order_[i];

            if(!fixed3Dregularizer_)
                views_[imgID]->computeSpatialRegularizer(sigma_p_);
            else
                views_[imgID]->update_k(sigma_p_,med_scene_depth_);

            // reset matches
            matches_[imgID] = std::vector<std::list<L3DPP::Match> >(views_[imgID]->num_lines());
            num_matches_[imgID] = 0;
            processed_[imgID] = false;
        }

        // find visual neighbors
        std::cout << prefix_ << "computing visual neighbors...     [" << num_neighbors_ << " imgs.]" << std::endl;
        std::cout << prefix_ << "starting to match " << views_.size() << " images..." << std::endl;

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<view_order_.size(); ++i)
        {
            unsigned int imgID = view_order_[i];

            if(fixed_visual_neighbors_.find(imgID) != fixed_visual_neighbors_.end())
            {
                if(visual_neighbors_[imgID].size() == 0)
                {
                    // fixed neighbors
                    std::list<unsigned int>::iterator n_it = fixed_visual_neighbors_[imgID].begin();
                    for(; n_it!=fixed_visual_neighbors_[imgID].end(); ++n_it)
                    {
                        if(views_.find(*n_it) != views_.end())
                            visual_neighbors_[imgID].insert(*n_it);
                    }
                }
            }
            else
            {
                // compute neighbors from WP overlap
//                findVisualNeighborsFromWPs(imgID);
                findVisualNeighborsFromROIs(imgID);
            }
        }

        // match images
        std::cout << prefix_ << "computing matches..." << std::endl;

        computeMatches();

        // translate back
        untranslate();

        /*
        std::ofstream fout;
        fout.open("/media/psf/Home/Desktop/lines.txt");
        std::vector<std::pair<L3DPP::Segment3D,L3DPP::Match> >::iterator iter =  estimated_position3D_.begin();
        for (; iter != estimated_position3D_.end(); iter++)
        {
            L3DPP::Segment3D seg3D = (*iter).first;
            fout << seg3D.P1().x() << " " << seg3D.P1().y() << " " << seg3D.P1().z() << " "
                 << seg3D.P2().x() << " " << seg3D.P2().y() << " " << seg3D.P2().z() << std::endl;
        }
        fout.close();
         */

        view_mutex_.unlock();
        view_reserve_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::translate()
    {
        if(views_.size() == 0)
            return;

        // find median x,y,z coordinates
        std::vector<std::vector<double> > coords(3);
        translation_ = Eigen::Vector3d(0,0,0);

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<3; ++i)
        {
            std::map<unsigned int,L3DPP::View*>::const_iterator it = views_.begin();
            for(; it!=views_.end(); ++it)
            {
                double val = (it->second->C())(i);
                if(fabs(val) > L3D_EPS)
                    coords[i].push_back(val);
            }

            if(coords[i].size() > 0)
            {
                std::sort(coords[i].begin(),coords[i].end());
                translation_(i) = coords[i][coords[i].size()/2];
            }
        }

        std::cout << prefix_ << "translation: ";
        std::cout << -translation_(0) << " ";
        std::cout << -translation_(1) << " ";
        std::cout << -translation_(2) << std::endl;

        // apply translation to views and 3D lines
        performTranslation(-translation_);
    }

    //------------------------------------------------------------------------------
    void Line3D::untranslate()
    {
        std::cout << prefix_ << "translating back..." << std::endl;

        // untranslate back to the original coordinates
        performTranslation(translation_);
    }

    //------------------------------------------------------------------------------
    void Line3D::performTranslation(const Eigen::Vector3d t)
    {
        // translate views
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<view_order_.size(); ++i)
        {
            views_[view_order_[i]]->translate(t);
        }

        // translate available 3D lines
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<lines3D_.size(); ++i)
        {
            L3DPP::FinalLine3D L = lines3D_[i];
            std::list<L3DPP::Segment3D>::iterator it = L.collinear3Dsegments_.begin();
            for(; it!=L.collinear3Dsegments_.end(); ++it)
            {
                (*it).translate(t);
            }

            L.underlyingCluster_.translate(t);
            lines3D_[i] = L;
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::findVisualNeighborsFromWPs(const unsigned int imgID)
    {
        if(visual_neighbors_.find(imgID) != visual_neighbors_.end())
        {
            // reset
            visual_neighbors_[imgID].clear();
            // views who can see common 3D points with this image, [imageID, common 3D point numbers]
            std::map<unsigned int,unsigned int> commonWPs;

            std::list<unsigned int>::const_iterator wp_it = views2worldpoints_[imgID].begin();
            for(; wp_it!=views2worldpoints_[imgID].end(); ++wp_it)
            {
                // iterate over worldpoints
                unsigned int wpID = *wp_it;

                std::list<unsigned int>::const_iterator view_it = worldpoints2views_[wpID].begin();
                for(; view_it!=worldpoints2views_[wpID].end(); ++view_it)
                {
                    // all views are potential neighbors
                    unsigned int vID = *view_it;
                    if(vID != imgID)
                    {
                        if(commonWPs.find(vID) == commonWPs.end())
                        {
                            commonWPs[vID] = 1;
                        }
                        else
                        {
                            ++commonWPs[vID];
                        }
                    }
                }
            }

            if(commonWPs.size() == 0)
                return;

            // find visual neighbors
            std::list<L3DPP::VisualNeighbor> neighbors;
            L3DPP::View* v = views_[imgID];
            std::map<unsigned int,unsigned int>::const_iterator c_it = commonWPs.begin();
            for(; c_it!=commonWPs.end(); ++c_it)
            {
                unsigned int vID = c_it->first;
                unsigned int num_common_wps = c_it->second;

                VisualNeighbor vn;
                vn.imgID_ = vID;
                vn.score_ = 2.0f*float(num_common_wps)/float(num_worldpoints_[imgID]+num_worldpoints_[vID]);
                vn.axisAngle_ = v->opticalAxesAngle(views_[vID]);  // optical axe angle of v and vID
                vn.distance_score_ = v->distanceVisualNeighborScore(views_[vID]); // block distance of optical axe of v and vID

                // check baseline
                if(vn.axisAngle_ < 1.571f && num_common_wps > 4) // ~ PI/2
                {
                    neighbors.push_back(vn);
                }
            }

            // sort by score
            neighbors.sort(L3DPP::sortVisualNeighborsByScore);

            // reduce to best neighbors
            if(neighbors.size() > num_neighbors_)
            {
                // copy neighbors
                std::list<L3DPP::VisualNeighbor> neighbors_tmp = neighbors;

                // get max score
                float score_t = 0.80f*neighbors.front().score_;
                unsigned int num_bigger_t = 0;

                // count the number of highly similar views
                std::list<L3DPP::VisualNeighbor>::const_iterator nit = neighbors.begin();
                while(nit!=neighbors.end() && (*nit).score_ > score_t)
                {
                    ++num_bigger_t;
                    ++nit;
                }

                neighbors.resize(num_bigger_t); // list.resize(n), only reserve element0, element1, ..., elementn

                // resort based on projective_score and world_point_score
                neighbors.sort(L3DPP::sortVisualNeighborsByDistScore);

                if(neighbors.size() > num_neighbors_/2)
                    neighbors.resize(num_neighbors_/2);

                // combine, add the new gotten neighbors in the front of original neighbors
//                std::cout << "lightol, neighbors.size() is " << neighbors.size() << std::endl;
                neighbors.splice(neighbors.end(),neighbors_tmp);
//                std::cout << "lightol, neighbors.size() is " << neighbors.size() << std::endl;
//                int a = 1;
            }



            // highscore neighbors -> store in visual neighbor map
//            volatile float a = v->getSpecificSpatialReg(0.5f);
//            volatile float b = v->initial_median_depth();
            float min_baseline = v->getSpecificSpatialReg(0.5f)*v->initial_median_depth();
            min_baseline = 0.1f;
            std::set<unsigned int> used_neighbors;  // store IDs of qualified neighbors
            std::list<L3DPP::VisualNeighbor>::const_iterator nit = neighbors.begin();
            while(nit!=neighbors.end() && used_neighbors.size() < num_neighbors_)
            {
                L3DPP::VisualNeighbor vn = *nit;
                L3DPP::View* v2 = views_[vn.imgID_];

                // check baseline
                // v->baseLine(v2) is the baseline length of this two images
                if(used_neighbors.find(vn.imgID_) == used_neighbors.end() && v->baseLine(v2) > min_baseline)
                {
                    std::set<unsigned int>::const_iterator u_it = used_neighbors.begin();
                    bool valid = true;
                    for(; u_it!=used_neighbors.end() && valid; ++u_it)
                    {
                        if(!(v->baseLine(views_[*u_it]) > min_baseline))
                            valid = false;
                    }

                    if(valid)
                        used_neighbors.insert(vn.imgID_);
                }

                ++nit;
            }

            visual_neighbors_[imgID] = used_neighbors;
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::computeMatches()
    {
        std::map<unsigned int,std::set<unsigned int> >::const_iterator it = visual_neighbors_.begin();
        for(; it!=visual_neighbors_.end(); ++it)
        {
            std::cout << prefix_;
            if(useGPU_)
                std::cout << "@GPU: ";
            else
                std::cout << "@CPU: " << std::endl;

            std::cout << "for view " << "[" << std::setfill('0') << std::setw(L3D_DISP_CAMS) << it->first << "], match it and its neighbor";

            // init GPU data
            if(useGPU_)
                initSrcDataGPU(it->first);

            // IDs of images who is neighbor of *it
            std::set<unsigned int>::const_iterator n_it = it->second.begin();
            for(; n_it!=it->second.end(); ++n_it)
            {
                if(matched_[it->first].find(*n_it) == matched_[it->first].end())
                {
                    // these two images have not yet do match
                    std::cout << "[" << std::setfill('0') << std::setw(L3D_DISP_CAMS) << *n_it << "], ";

                    // compute fundamental matrix
                    Eigen::Matrix3d F = getFundamentalMatrix(views_[it->first],
                                                             views_[*n_it]);

                    // matching
                    if(useGPU_)
                        matchingGPU(it->first,*n_it,F);
                    else
                        matchingCPU(it->first,*n_it,F);

                    // set matched
                    matched_[it->first].insert(*n_it);
                    matched_[*n_it].insert(it->first);
                }
            }

            std::cout << "done!" << std::endl;

            // check matches for orientation
            if(L3D_DEF_CHECK_MATCH_ORIENTATION)
            {
                checkMatchOrientation(it->first);
            }

            // scoring
            float valid_f;

            if(useGPU_)
                scoringGPU(it->first,valid_f);
            else
                // compute score3D_ for all match in matches
                scoringCPU(it->first,valid_f);  // use angle and distance to prune wrong line segment matches

            std::cout << prefix_ << "scoring: " << "clusterable_segments = " << int(valid_f*100) << "%";
            std::cout << std::endl;

            // cleanup GPU data
            if(useGPU_)
                removeSrcDataGPU(it->first);

            // store inverse matches, for matches(i, j), add according matches(j, i)
            // TODO::it is important because when matchCPU we only unproject src segments to tgt image plane!
            storeInverseMatches(it->first);

            // filter invalid matches
            // for segments who has match.score3D_ > threshold, generate the estimated 3D position,
            // filter all matches whose score3D_ is less than 0.1*threshold
            filterMatches(it->first);

            // set processed
            processed_[it->first] = true;

            std::cout << prefix_ << "#matches: ";
            std::cout << std::setfill(' ') << std::setw(L3D_DISP_MATCHES) << num_matches_[it->first] << std::endl;
            std::cout << prefix_ << "median_depth: " << views_[it->first]->median_depth() << std::endl;
        }

        /*
        // DEBUG: save all remaining matches
        std::vector<L3DPP::Segment3D> all_matches;
        std::map<unsigned int,std::vector<std::list<L3DPP::Match> > >::iterator dbg_it = matches_.begin();
        for(; dbg_it!=matches_.end(); ++dbg_it)
        {
            L3DPP::View* v = views_[dbg_it->first];
            for(size_t i=0; i<dbg_it->second.size(); ++i)
            {
                std::list<L3DPP::Match>::iterator dbg_it2 = dbg_it->second.at(i).begin();
                for(; dbg_it2!=dbg_it->second.at(i).end(); ++dbg_it2)
                {
                    L3DPP::Match m = *dbg_it2;
                    L3DPP::Segment3D seg3D = v->unprojectSegment(m.src_segID_,m.depth_p1_,m.depth_p2_);
                    all_matches.push_back(seg3D);
                }
            }
        }
        saveTempResultAsSTL(data_folder_,"all",all_matches);

        // DEBUG: save best hypotheses
        std::vector<L3DPP::Segment3D> best_matches;
        for(size_t i=0; i<estimated_position3D_.size(); ++i)
        {
            best_matches.push_back(estimated_position3D_[i].first);
        }
        saveTempResultAsSTL(data_folder_,"best",best_matches);
        */
    }

    //------------------------------------------------------------------------------
    void Line3D::checkMatchOrientation(const unsigned int src)
    {
        if(matches_.find(src) == matches_.end())
            return;

        unsigned int num_matches_before = num_matches_[src];
        unsigned int num_matches = 0;
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP

        //todo:: here, output without any filtering


        for(size_t i=0; i<matches_[src].size(); ++i)  // for each line segment in source view
        {
            std::list<L3DPP::Match> remaining;

            std::list<L3DPP::Match>::const_iterator it = matches_[src][i].begin();
            int size = matches_[src][i].size();
            for(; it!=matches_[src][i].end(); ++it)
            {
                L3DPP::Match m = *it;

                // unproject
                L3DPP::Segment3D seg3D = unprojectMatch(m);  // get the line segment of src image in 3D space

                // check angle
                double ang = views_[m.src_camID_]->segmentQualityAngle(seg3D,m.src_segID_);

                if(ang > L3D_PI_1_32 && ang < L3D_PI_31_32)
                {
                    remaining.push_back(m);
                }
            }

            matches_[src][i] = remaining;

            match_mutex_.lock();
            num_matches += remaining.size();
            match_mutex_.unlock();
        }

        num_matches_[src] = num_matches;

        float perc = 0;
        if(num_matches_before > 0)
            perc = float(num_matches)/float(num_matches_before)*100.0f;

        std::cout << prefix_ << "filter matches by orientation... ";
        std::cout << num_matches_before << " --> " << num_matches;
        std::cout << " (~" << int(perc) << "%)" << std::endl;
    }

    //------------------------------------------------------------------------------
    Eigen::Matrix3d Line3D::getFundamentalMatrix(L3DPP::View* src, L3DPP::View* tgt)
    {
        // check if it already exists
        if(fundamentals_[src->id()].find(tgt->id()) != fundamentals_[src->id()].end())
        {
            return fundamentals_[src->id()][tgt->id()];
        }
        else if(fundamentals_[tgt->id()].find(src->id()) != fundamentals_[tgt->id()].end())
        {
            Eigen::Matrix3d Ft = fundamentals_[tgt->id()][src->id()].transpose();
            return Ft;
        }

        // compute new fundamental matrix
        Eigen::Matrix3d K1 = src->K();
        Eigen::Matrix3d R1 = src->R();
        Eigen::Vector3d t1 = src->t();

        Eigen::Matrix3d K2 = tgt->K();
        Eigen::Matrix3d R2 = tgt->R();
        Eigen::Vector3d t2 = tgt->t();

        Eigen::Matrix3d R = R2 * R1.transpose();
        Eigen::Vector3d t = t2 - R * t1;

        Eigen::Matrix3d T(3,3);
        T(0,0) = 0.0;    T(0,1) = -t.z(); T(0,2) = t.y();
        T(1,0) = t.z();  T(1,1) = 0.0;    T(1,2) = -t.x();
        T(2,0) = -t.y(); T(2,1) = t.x();  T(2,2) = 0.0;

        Eigen::Matrix3d E = T * R;
        Eigen::Matrix3d F = K2.transpose().inverse() * E * K1.inverse();

        fundamentals_[src->id()][tgt->id()] = F;

        return F;
    }

    //------------------------------------------------------------------------------
    void Line3D::matchingCPU(const unsigned int src, const unsigned int tgt,
                             const Eigen::Matrix3d& F)
    {
        L3DPP::View* v_src = views_[src];
        L3DPP::View* v_tgt = views_[tgt];
        std::vector<Eigen::Vector3d> centrals_src = v_src->centrals();
        std::vector<Eigen::Vector3d> centrals_tgt = v_tgt->centrals();
        // distinguish who is front and who is back
        Eigen::Vector3d a = centrals_src.front();
        Eigen::Vector3d b = centrals_src.back();
        Eigen::Vector3d c = centrals_tgt.front();
        Eigen::Vector3d d = centrals_tgt.back();
//        Eigen::Vector3d ca = c-a;
//        Eigen::Vector3d cb = c-b;

        // todo:: use the driving direction to judge who is front
        bool srcF = true;  // src image is in front
        if (src < tgt)
        {
            srcF = false;
        }

//        cout << "srcID is " << src << ", tgtID is " << tgt << endl;
//        for (int i = 0; i < centrals_src.size(); ++i)
//        {
//            cout << centrals_src[i].x() << std::setfill(' ') << " " << std::setw(10) << centrals_src[i].y() << " " << std::setw(10) << centrals_src[i].z() << " ";
//            cout << centrals_tgt[i].x() << std::setfill(' ') << " " << std::setw(10) << centrals_tgt[i].y() << " " << std::setw(10) << centrals_tgt[i].z() << endl;
//        }

        int num_lines = v_src->lines()->width();
        std::vector<double> ava_dist(num_lines-1);

        for (int i = 1; i < num_lines; ++i)
        {
            for (int j = 0; j < num_lines-i; ++j)
            {
                if (srcF)
                {
                    ava_dist[i-1] = ava_dist[i-1] + (centrals_src[j] - centrals_tgt[j+i]).norm();
                }
                else
                {
                    ava_dist[i-1] = ava_dist[i-1] + (centrals_tgt[j] - centrals_src[j+i]).norm();
                }

            }
            ava_dist[i-1] /= num_lines-i;
        }

        std::vector<double>::iterator it = std::min_element(ava_dist.begin(), ava_dist.end());
        double min_ava = *it;

        // todo:: here, choose a threshold
        if (min_ava > 1)
        {
            return;
        }

        int offset = std::distance(ava_dist.begin(), it) + 1;
        L3DPP::DataArray<float4>* lines_src = v_src->lines();
        L3DPP::DataArray<float4>* lines_tgt = v_tgt->lines();
        unsigned int num_matches = 0;   // matches number between all segments in src and tgt image

        for (size_t r=0; r < num_lines-offset; ++r)
        {
            int new_matches = 0;
            Eigen::Vector3d p1,p2,q1,q2;
            // use priority queue when kNN > 0
            // matches between segment r in source image and all segments in target image
            L3DPP::pairwise_matches scored_matches;
            int srcSegID = 0;
            int tgtSegID = 0;

            if (srcF)
            {
                srcSegID = r;
                tgtSegID = r+offset;
            }
            else
            {
                srcSegID = r+offset;
                tgtSegID = r;
            }

            // source line
            p1 = Eigen::Vector3d(lines_src->dataCPU(srcSegID,0)[0].x,
                                 lines_src->dataCPU(srcSegID,0)[0].y,1.0);
            p2 = Eigen::Vector3d(lines_src->dataCPU(srcSegID,0)[0].z,
                                 lines_src->dataCPU(srcSegID,0)[0].w,1.0);


            // target line
            q1 = Eigen::Vector3d(lines_tgt->dataCPU(tgtSegID,0)[0].x,
                                 lines_tgt->dataCPU(tgtSegID,0)[0].y,1.0);
            q2 = Eigen::Vector3d(lines_tgt->dataCPU(tgtSegID,0)[0].z,
                                 lines_tgt->dataCPU(tgtSegID,0)[0].w,1.0);

            // triangulate
            Eigen::Vector2d depths_src = triangulationDepths(src,p1,p2,
                                                             tgt,q1,q2);
            Eigen::Vector2d depths_tgt = triangulationDepths(tgt,q1,q2,
                                                             src,p1,p2);

            if(depths_src.x() > L3D_EPS && depths_src.y() > L3D_EPS &&
               depths_tgt.x() > L3D_EPS && depths_tgt.y() > L3D_EPS)
            {
                // potential match
                L3DPP::Match M;
                M.src_camID_ = src;
                M.src_segID_ = srcSegID;
                M.tgt_camID_ = tgt;
                M.tgt_segID_ = tgtSegID;
                M.overlap_score_ = 1;
                M.score3D_ = 0.0f;
                M.depth_p1_ = depths_src.x();
                M.depth_p2_ = depths_src.y();
                M.depth_q1_ = depths_tgt.x();
                M.depth_q2_ = depths_tgt.y();

                L3DPP::View* v = views_[src];
//                 todo::here, watch out!
                L3DPP::Segment3D seg3D = v->unprojectSegment(M.src_segID_,M.depth_p1_,M.depth_p2_);
                std::ofstream fout;
                fout.open("/media/psf/Home/Desktop/lines.txt", std::ios::app);
                fout << seg3D.P1().x() << " " << seg3D.P1().y() << " " << seg3D.P1().z() << " "
                     << seg3D.P2().x() << " " << seg3D.P2().y() << " " << seg3D.P2().z() << std::endl;
                fout.close();

                if(kNN_ > 0)  // only use  kNN_ matches at most
                {
                    // kNN matching
                    scored_matches.push(M);
                }
                else
                {
                    // all matches are used
                    matches_[src][r].push_back(M);
                    ++new_matches;
                }
            }


            // push kNN matches into list
//            volatile int tmp = scored_matches.size();
            int num = scored_matches.size();
            if(kNN_ > 0)
            {
                while(new_matches < kNN_ && !scored_matches.empty())
                {
                    matches_[src][srcSegID].push_back(scored_matches.top());
                    scored_matches.pop();
                    ++new_matches;
                }
            }

            match_mutex_.lock();
            num_matches += new_matches;
            match_mutex_.unlock();
        }
/*
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t r=0; r<lines_src->width(); ++r)
        {
            int new_matches = 0;

            // source line
//            std::cout << lines_src->dataCPU(r,0) << std::endl;
//            std::cout << lines_src->dataCPU(r,0)->x << std::endl;
            Eigen::Vector3d p1(lines_src->dataCPU(r,0)[0].x,
                               lines_src->dataCPU(r,0)[0].y,1.0);
            Eigen::Vector3d p2(lines_src->dataCPU(r,0)[0].z,
                               lines_src->dataCPU(r,0)[0].w,1.0);

            // epipolar lines
            Eigen::Vector3d epi_p1 = F*p1;  // with fundamental matrix and p, we can get the corresponding epipolar lines
            Eigen::Vector3d epi_p2 = F*p2;

            // use priority queue when kNN > 0
            // matches between segment r in source image and all segments in target image
            L3DPP::pairwise_matches scored_matches;

            for(size_t c=0; c<lines_tgt->width(); ++c)
            {
                // target line
                Eigen::Vector3d q1(lines_tgt->dataCPU(c,0)[0].x,
                                   lines_tgt->dataCPU(c,0)[0].y,1.0);
                Eigen::Vector3d q2(lines_tgt->dataCPU(c,0)[0].z,
                                   lines_tgt->dataCPU(c,0)[0].w,1.0);
                Eigen::Vector3d l2 = q1.cross(q2);

                // intersect
                // todo:: here, something is wrong, cross multiple will cause errors
//                Eigen::Vector3d p1_proj = l2.cross(epi_p1);
//                Eigen::Vector3d p2_proj = l2.cross(epi_p2);
                Eigen::Vector3d p1_proj;
                p1_proj(0) = -(epi_p1(1)*q1(1) + epi_p1(2))/epi_p1(0);
                p1_proj(1) = q1(1);
                p1_proj(2) = 1;
                Eigen::Vector3d p2_proj;
                p2_proj(0) = -(epi_p2(1)*q1(1) + epi_p2(2))/epi_p2(0);
                p2_proj(1) = q1(1);
                p2_proj(2) = 1;
//                Eigen::Vector3d p2_proj = l2.cross(epi_p2);


                if(fabs(p1_proj.z()) > L3D_EPS && fabs(p2_proj.z()) > L3D_EPS)
                {
                    // normalize
                    p1_proj /= p1_proj.z();
                    p2_proj /= p2_proj.z();

                    // check overlap
                    std::vector<Eigen::Vector3d> collinear_points(4);
                    collinear_points[0] = p1_proj;
                    collinear_points[1] = p2_proj;
                    collinear_points[2] = q1;
                    collinear_points[3] = q2;
                    float score = mutualOverlap(collinear_points);

                    if(score > epipolar_overlap_)
                    {
                        // triangulate
                        Eigen::Vector2d depths_src = triangulationDepths(src,p1,p2,
                                                                         tgt,q1,q2);
                        Eigen::Vector2d depths_tgt = triangulationDepths(tgt,q1,q2,
                                                                         src,p1,p2);

                        if(depths_src.x() > L3D_EPS && depths_src.y() > L3D_EPS &&
                           depths_tgt.x() > L3D_EPS && depths_tgt.y() > L3D_EPS &&
                           depths_src.x() < 30 && depths_src.y() < 30 &&
                           depths_tgt.x() < 30 && depths_tgt.y() < 30)
                        {
                            // potential match
                            L3DPP::Match M;
                            M.src_camID_ = src;
                            M.src_segID_ = r;
                            M.tgt_camID_ = tgt;
                            M.tgt_segID_ = c;
                            M.overlap_score_ = score;
                            M.score3D_ = 0.0f;
                            M.depth_p1_ = depths_src.x();
                            M.depth_p2_ = depths_src.y();
                            M.depth_q1_ = depths_tgt.x();
                            M.depth_q2_ = depths_tgt.y();

                            if(kNN_ > 0)
                            {
                                // kNN matching
                                scored_matches.push(M);
                            }
                            else
                            {
                                // all matches are used
                                matches_[src][r].push_back(M);
                                ++new_matches;
                            }
                        }
                    }
                }
            }

            // push kNN matches into list
//            volatile int tmp = scored_matches.size();
            if(kNN_ > 0)
            {
                while(new_matches < kNN_ && !scored_matches.empty())
                {
                    matches_[src][r].push_back(scored_matches.top());
                    scored_matches.pop();
                    ++new_matches;
                }
            }

            match_mutex_.lock();
            num_matches += new_matches;
            match_mutex_.unlock();
        }
        */

        num_matches_[src] += num_matches;
    }

    //------------------------------------------------------------------------------
    void Line3D::initSrcDataGPU(const unsigned int src)
    {
#ifdef L3DPP_CUDA
        // upload
        L3DPP::View* v1 = views_[src];
        v1->lines()->upload();
        v1->RtKinvGPU()->upload();
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    void Line3D::removeSrcDataGPU(const unsigned int src)
    {
#ifdef L3DPP_CUDA
        // cleanup
        L3DPP::View* v1 = views_[src];
        v1->lines()->removeFromGPU();
        v1->RtKinvGPU()->removeFromGPU();
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    void Line3D::matchingGPU(const unsigned int src, const unsigned int tgt,
                             const Eigen::Matrix3d& F)
    {
#ifdef L3DPP_CUDA
        // INFO: src data must be on GPU! initSrcDataGPU(src)
        L3DPP::View* v1 = views_[src];

        // upload segments to GPU
        L3DPP::View* v2 = views_[tgt];
        v2->lines()->upload();

        // move F to GPU
        L3DPP::DataArray<float>* F_GPU = NULL;
        eigen2dataArray(F_GPU,F);
        F_GPU->upload();

        // move RtKinv to GPU
        v2->RtKinvGPU()->upload();

        // match segments on GPU
        unsigned int num_matches = L3DPP::match_lines_GPU(v1->lines(),v2->lines(),F_GPU,
                                                          v1->RtKinvGPU(),v2->RtKinvGPU(),
                                                          v1->C_GPU(),v2->C_GPU(),
                                                          &(matches_[src]),src,tgt,
                                                          epipolar_overlap_,kNN_);

        num_matches_[src] += num_matches;

        // cleanup
        v2->lines()->removeFromGPU();
        v2->RtKinvGPU()->removeFromGPU();
        delete F_GPU;

#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    bool Line3D::pointOnSegment(const Eigen::Vector3d& x, const Eigen::Vector3d& p1,
                                const Eigen::Vector3d& p2)
    {
        Eigen::Vector2d v1(p1.x()-x.x(),p1.y()-x.y());
        Eigen::Vector2d v2(p2.x()-x.x(),p2.y()-x.y());
        return (v1.dot(v2) < L3D_EPS);  // judge if v1,v2 is the same direction, i.e., x is not in the
                                        // middle of x and y
    }

    //------------------------------------------------------------------------------
    float Line3D::mutualOverlap(const std::vector<Eigen::Vector3d>& collinear_points)
    {
        float overlap = 0.0f;

        if(collinear_points.size() != 4)
            return 0.0f;

        Eigen::Vector3d p1 = collinear_points[0];
        Eigen::Vector3d p2 = collinear_points[1];
        Eigen::Vector3d q1 = collinear_points[2];
        Eigen::Vector3d q2 = collinear_points[3];

        if(pointOnSegment(p1,q1,q2) || pointOnSegment(p2,q1,q2) ||
                pointOnSegment(q1,p1,p2) || pointOnSegment(q2,p1,p2))
        {
            // find outer distance and inner points
            float max_dist = 0.0f;
            size_t outer1 = 0;
            size_t inner1 = 1;
            size_t inner2 = 2;
            size_t outer2 = 3;

            // find the largest interval
            for(size_t i=0; i<3; ++i)
            {
                for(size_t j=i+1; j<4; ++j)
                {
//                    const volatile Eigen::Vector3d tmp = collinear_points[i]-collinear_points[j];
//                    const volatile double tmpDist = sqrt(tmp(0)*tmp(0) + tmp(1)*tmp(1) + tmp(2)*tmp(2));
                    float dist = (collinear_points[i]-collinear_points[j]).norm();
                    if(dist > max_dist)
                    {
                        max_dist = dist;
                        outer1 = i;
                        outer2 = j;
                    }
                }
            }

            if(max_dist < 1.0f)
                return 0.0f;

            if(outer1 == 0)
            {
                if(outer2 == 1)
                {
                    inner1 = 2;
                    inner2 = 3;
                }
                else if(outer2 == 2)
                {
                    inner1 = 1;
                    inner2 = 3;
                }
                else
                {
                    inner1 = 1;
                    inner2 = 2;
                }
            }
            else if(outer1 == 1)
            {
                inner1 = 0;
                if(outer2 == 2)
                {
                    inner2 = 3;
                }
                else
                {
                    inner2 = 2;
                }
            }
            else
            {
                inner1 = 0;
                inner2 = 1;
            }

            overlap = (collinear_points[inner1]-collinear_points[inner2]).norm()/max_dist;

        }
        return overlap;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector2d Line3D::triangulationDepths(const unsigned int src_camID, const Eigen::Vector3d& p1,
                                                const Eigen::Vector3d& p2, const unsigned int tgt_camID,
                                                const Eigen::Vector3d& line_q1, const Eigen::Vector3d& line_q2)
    {
        L3DPP::View* v_src = views_[src_camID];
        L3DPP::View* v_tgt = views_[tgt_camID];

        // rays through points
        Eigen::Vector3d C1 = v_src->C();
        Eigen::Vector3d ray_p1 = v_src->getNormalizedRay(p1);
        Eigen::Vector3d ray_p2 = v_src->getNormalizedRay(p2);


        // plane
        Eigen::Vector3d C2 = v_tgt->C();
        Eigen::Vector3d ray_q1 = v_tgt->getNormalizedRay(line_q1);
        Eigen::Vector3d ray_q2 = v_tgt->getNormalizedRay(line_q2);
        Eigen::Vector3d n = ray_q1.cross(ray_q2);
        n.normalize();

        using std::cout;
        using std::endl;
        /*
        cout << "R is " << v_src->R() << endl;
        cout << "t is " << v_src->t() << endl;
        cout << "C1 is " << C1 << endl;
        cout << "C2 is " << C2 << endl;
        cout << "p1 is " << p1 << endl;
        cout << "p2 is " << p2 << endl;
        cout << "q1 is " << line_q1 << endl;
        cout << "q2 is " << line_q2 << endl;
        cout << "ray_p1 is " << ray_p1 << endl;
        cout << "ray_p2 is " << ray_p2 << endl;
        cout << "ray_q1 is " << ray_q1 << endl;
        cout << "ray_q2 is " << ray_q2 << endl;
        cout << "n is " << n << endl;
         */


        // the p1p2 is exactly on the epipolar line
        if(fabs(ray_p1.dot(n)) < L3D_EPS || fabs(ray_p2.dot(n)) < L3D_EPS)
            return Eigen::Vector2d(-1,-1);

        volatile float tmp1 = n.dot(ray_p1);
        volatile float tmp2 = n.dot(ray_p2);

        // if R and t is right, d1 and d2 should be > 0
        double d1 = (C2.dot(n) - n.dot(C1)) / (n.dot(ray_p1));
        double d2 = (C2.dot(n) - n.dot(C1)) / (n.dot(ray_p2));
        return Eigen::Vector2d(d1,d2);
    }

    //------------------------------------------------------------------------------
    void Line3D::sortMatches(const unsigned int src)
    {
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<matches_[src].size(); ++i)
        {
            matches_[src][i].sort(L3DPP::sortMatchesByIDs);
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::scoringCPU(const unsigned int src, float& valid_f)
    {
        // init
        valid_f = 0.0f;
        L3DPP::View* v = views_[src];
        float k = v->k();  // unstability coefficient of this image

        unsigned int num_valid = 0;  // how many reliable matched segment exist in this image

        // iterative scoring
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<matches_[src].size(); ++i)  // for each line segment in src view
        {
            bool valid_match_exists = false;  // if reliable matched segment exist in this image

            std::list<L3DPP::Match>::iterator it = matches_[src][i].begin();
            for(; it!=matches_[src][i].end(); ++it)
            {
                L3DPP::Match M = *it;
                float score3D = 0.0f;  // the accumulated similarity of this match in all neighbors
                std::map<unsigned int,float> score_per_cam;

                // unproject once
                L3DPP::Segment3D M3D = v->unprojectSegment(M.src_segID_,M.depth_p1_,M.depth_p2_);

                // compute spatial regularizers
                float reg1,reg2;
                float sig1 = M.depth_p1_*k;
                float sig2 = M.depth_p2_*k;

                reg1 = 2.0f*sig1*sig1;
                reg2 = 2.0f*sig2*sig2;

                // compute spatial regularizers (tgt)
                //TODO:: why dont here use M.depth_q1_, M.depth_q2_
                // ans:: it seems equal
                float sig1_tgt = views_[M.tgt_camID_]->regularizerFrom3Dpoint(M3D.P1());
                float sig2_tgt = views_[M.tgt_camID_]->regularizerFrom3Dpoint(M3D.P2());

                reg1 = 0.5f*(reg1 + 2.0f*sig1_tgt*sig1_tgt);
                reg2 = 0.5f*(reg2 + 2.0f*sig2_tgt*sig2_tgt);

                std::list<L3DPP::Match>::const_iterator it2 = matches_[src][i].begin();
                for(; it2!=matches_[src][i].end(); ++it2)
                {
                    L3DPP::Match M2 = *it2;

                    if(M.tgt_camID_ != M2.tgt_camID_)
                    {
                        // compute similarity of match M and M2, sim must be the highest score of all seg matches in src and tgt
                        // image, too easy to understand! one segment will not have two matched segs in the same tge image!
                        float sim = similarityForScoring(M,M2,M3D,reg1,reg2);

                        if(score_per_cam.find(M2.tgt_camID_) != score_per_cam.end())
                        {
                            if(sim > score_per_cam[M2.tgt_camID_])
                            {
                                //one target image will only contribute one seg to support this match
                                score3D -= score_per_cam[M2.tgt_camID_];
                                score3D += sim;
                                score_per_cam[M2.tgt_camID_] = sim;
                            }
                        }
                        else
                        {
                            score3D += sim;
                            score_per_cam[M2.tgt_camID_] = sim;
                        }
                    }
                }

                (*it).score3D_ = score3D;
                if(score3D > L3D_DEF_MIN_BEST_SCORE_3D)
                {
                    valid_match_exists = true;
                }
            }

            if(valid_match_exists)
            {
                scoring_mutex_.lock();
                ++num_valid;
                scoring_mutex_.unlock();
            }
        }

        // check number of segments with valid matches
        valid_f = float(num_valid)/float(v->num_lines());
    }

    //------------------------------------------------------------------------------
    void Line3D::scoringGPU(const unsigned int src, float& valid_f)
    {
#ifdef L3DPP_CUDA
        // INFO: src data must be on GPU! initSrcDataGPU(src) -> remove afterwards!

        // init
        valid_f = 0.0f;
        L3DPP::View* v = views_[src];
        float k = v->k();

        if(num_matches_[src] == 0)
            return;

        // sort matches by ids first
        sortMatches(src);

        // find start and end indices
        L3DPP::DataArray<int2>* ranges = new L3DPP::DataArray<int2>(v->num_lines(),1);
        unsigned int offset = 0;
        for(size_t i=0; i<v->num_lines(); ++i)
        {
            if(matches_[src][i].size() > 0)
            {
                ranges->dataCPU(i,0)[0] = make_int2(offset,offset+matches_[src][i].size()-1);
                offset += matches_[src][i].size();
            }
            else
            {
                // no matches for this segment
                ranges->dataCPU(i,0)[0] = make_int2(-1,-1);
            }
        }

        // store matches in array
        L3DPP::DataArray<float4>* matches = new L3DPP::DataArray<float4>(num_matches_[src],1);
        L3DPP::DataArray<float2>* regularizers_tgt = new L3DPP::DataArray<float2>(num_matches_[src],1);
        L3DPP::DataArray<float>* scores = new L3DPP::DataArray<float>(num_matches_[src],1,true);

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<matches_[src].size(); ++i)
        {
            int offset = ranges->dataCPU(i,0)[0].x;
            if(offset >= 0)
            {
                int id = 0;
                std::list<L3DPP::Match>::const_iterator it = matches_[src][i].begin();
                for(; it!=matches_[src][i].end(); ++it,++id)
                {
                    L3DPP::Match m = *it;
                    matches->dataCPU(offset+id,0)[0] = make_float4(i,m.tgt_camID_,
                                                                   m.depth_p1_,m.depth_p2_);
                    L3DPP::Segment3D s3D = v->unprojectSegment(m.src_segID_,m.depth_p1_,m.depth_p2_);
                    regularizers_tgt->dataCPU(offset+id,0)[0] = make_float2(views_[m.tgt_camID_]->regularizerFrom3Dpoint(s3D.P1()),
                                                                            views_[m.tgt_camID_]->regularizerFrom3Dpoint(s3D.P2()));
                }
            }
        }

        // upload
        ranges->upload();
        matches->upload();
        regularizers_tgt->upload();

        unsigned int num_valid = 0;

        // score on GPU
        L3DPP::score_matches_GPU(v->lines(),matches,ranges,scores,regularizers_tgt,
                                 v->RtKinvGPU(),v->C_GPU(),
                                 two_sigA_sqr_,k,L3D_DEF_MIN_SIMILARITY_3D);
        scores->download();

        // write back
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<matches_[src].size(); ++i)
        {
            bool valid_match_exists = false;
            int offset = ranges->dataCPU(i,0)[0].x;
            if(offset >= 0)
            {
                int id = 0;
                std::list<L3DPP::Match>::iterator it = matches_[src][i].begin();
                for(; it!=matches_[src][i].end(); ++it,++id)
                {
                    // get score
                    float score = scores->dataCPU(offset+id,0)[0];

                    // update
                    (*it).score3D_ = score;

                    if(score > L3D_DEF_MIN_BEST_SCORE_3D)
                    {
                        valid_match_exists = true;
                    }
                }
            }

            if(valid_match_exists)
            {
                scoring_mutex_.lock();
                ++num_valid;
                scoring_mutex_.unlock();
            }
        }

        // check number of segments with valid matches
        valid_f = float(num_valid)/float(v->num_lines());

        // cleanup
        delete ranges;
        delete matches;
        delete scores;
        delete regularizers_tgt;
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    float Line3D::similarityForScoring(const L3DPP::Match& m1, const L3DPP::Match& m2,
                                       const L3DPP::Segment3D& seg3D1,
                                       const float reg1, const float reg2)
    {
        // todo:: loose the constraint, cause we cannot garantee that endpoints meet well
        // todo:: a new method computing similarity must be applied here
        float Reg1 = reg1*100;
        float Reg2 = reg2*100;
        L3DPP::Segment3D seg3D2 = unprojectMatch(m2,true);

        if(seg3D1.length() < L3D_EPS || seg3D2.length() < L3D_EPS)
            return 0.0f;

        // angular similarity
        float angle = angleBetweenSeg3D(seg3D1,seg3D2,true);
        float sim_a = expf(-angle*angle/two_sigA_sqr_);

        // positional similarity
        float sim_p = 0.0f;
        if(m1.src_camID_ == m2.src_camID_ && m1.src_segID_ == m2.src_segID_)
        {
            // local similarity
            float d1 = m1.depth_p1_-m2.depth_p1_;
            float d2 = m1.depth_p2_-m2.depth_p2_;

            volatile float tmp1 = expf(-d1*d1/Reg1);
            volatile float tmp2 = expf(-d2*d2/Reg2);

//            volatile float tmp3 = expf(-d1*d1/reg1);
//            volatile float tmp4 = expf(-d2*d2/reg2);

            sim_p = fmin(expf(-d1*d1/Reg1),expf(-d2*d2/Reg2));
        }

//        float sim = fmin(sim_a,sim_p);
        float sim = sim_a;  // todo:: sim_p can also provide something
        if(sim > L3D_DEF_MIN_SIMILARITY_3D)
            return sim;
        else
            return 0.0f;
    }

    //------------------------------------------------------------------------------
    float Line3D::similarity(const L3DPP::Segment2D& seg1, const L3DPP::Segment2D& seg2,
                             const bool truncate)
    {
        // check for 3D estimates
        if(entry_map_.find(seg1) == entry_map_.end())
        {
            return 0.0f;
        }

        size_t ent1 = entry_map_[seg1];
        std::pair<L3DPP::Segment3D,L3DPP::Match> data1 = estimated_position3D_[ent1];
        L3DPP::Segment3D s1 = data1.first;
        L3DPP::Match m1 = data1.second;

        return similarity(s1,m1,seg2,truncate);
    }

    //------------------------------------------------------------------------------
    float Line3D::similarity(const L3DPP::Segment3D& s1, const L3DPP::Match& m1,
                             const L3DPP::Segment2D& seg2, const bool truncate)
    {
        // check for 3D estimates
        // if seg2 also have best_matches, i.e., it is supported by many matches, then continue similarity it
        if(entry_map_.find(seg2) == entry_map_.end())
        {
            return 0.0f;
        }

        size_t ent2 = entry_map_[seg2];
        std::pair<L3DPP::Segment3D,L3DPP::Match> data2 = estimated_position3D_[ent2];
        L3DPP::Segment3D s2 = data2.first;
        L3DPP::Match m2 = data2.second;

        if(s1.length() < L3D_EPS || s2.length() < L3D_EPS)
            return 0.0f;

        L3DPP::View* v1 = views_[m1.src_camID_];
        L3DPP::View* v2 = views_[m2.src_camID_];

        // angular similarity
        float angle = angleBetweenSeg3D(s1,s2,true);
        float sim_a = expf(-angle*angle/two_sigA_sqr_);

        // cutoff depths
        float cutoff1 = v1->median_depth();
        float cutoff2 = v2->median_depth();

        if(med_scene_depth_lines_ > L3D_EPS)
        {
            cutoff1 = fmin(cutoff1,med_scene_depth_lines_);
            cutoff2 = fmin(cutoff2,med_scene_depth_lines_);
        }

        // positional similarity
        float d11 = s2.distance_Point2Line(s1.P1());
        float d12 = s2.distance_Point2Line(s1.P2());
        float d21 = s1.distance_Point2Line(s2.P1());
        float d22 = s1.distance_Point2Line(s2.P2());

        float reg11,reg12,reg21,reg22;
        float sig11;
        // every depth will be assigned a value as small as possible, to garantee accuracy
        if(m1.depth_p1_ > cutoff1)
            sig11 = cutoff1*v1->k();
        else
            sig11 = m1.depth_p1_*v1->k();

        float sig12;
        if(m1.depth_p2_ > cutoff1)
            sig12 = cutoff1*v1->k();
        else
            sig12 = m1.depth_p2_*v1->k();

        reg11 = 2.0f*sig11*sig11;
        reg12 = 2.0f*sig12*sig12;

        float sig21;
        if(m2.depth_p1_ > cutoff2)
            sig21 = cutoff2*v2->k();
        else
            sig21 = m2.depth_p1_*v2->k();

        float sig22;
        if(m2.depth_p2_ > cutoff2)
            sig22 = cutoff2*v2->k();
        else
            sig22 = m2.depth_p2_*v2->k();

        reg21 = 2.0f*sig21*sig21;
        reg22 = 2.0f*sig22*sig22;

        float sim_p1 = fmin(expf(-d11*d11/reg11),expf(-d12*d12/reg12));
        float sim_p2 = fmin(expf(-d21*d21/reg21),expf(-d22*d22/reg22));

        float sim_p = fmin(sim_p1,sim_p2);

        float sim = fmin(sim_a,sim_p);

        if(truncate)
        {
            if(sim > L3D_DEF_MIN_SIMILARITY_3D)
                return sim;
            else
                return 0.0f;
        }
        return sim;
    }

    //------------------------------------------------------------------------------
    L3DPP::Segment3D Line3D::unprojectMatch(const L3DPP::Match& m, const bool src)
    {
        if(src)
        {
            L3DPP::View* v = views_[m.src_camID_];
            // todo::here, watch out!
            return v->unprojectSegment(m.src_segID_,m.depth_p1_,m.depth_p2_);
        }
        else
        {
            L3DPP::View* v = views_[m.tgt_camID_];
            return v->unprojectSegment(m.tgt_segID_,m.depth_q1_,m.depth_q2_);
        }
    }

    //------------------------------------------------------------------------------
    float Line3D::angleBetweenSeg3D(const L3DPP::Segment3D& s1, const L3DPP::Segment3D& s2,
                                    const bool undirected)
    {
        float dot_p = s1.dir().dot(s2.dir());
        float angle = acos(fmax(fmin(dot_p,1.0f),-1.0f))/M_PI*180.0f;

        if(undirected && angle > 90.0f)
        {
            angle = 180.0f-angle;
        }

        return angle;
    }

    //------------------------------------------------------------------------------
    void Line3D::filterMatches(const unsigned int src)
    {
        // filter and find median depth
        std::vector<float> depths;

        // compute maximum score for this view
        float max_score = 0.0f;
        for(size_t i=0; i<matches_[src].size(); ++i)  // for each line segment
        {
            std::list<L3DPP::Match>::const_iterator it = matches_[src][i].begin();
            for(; it!=matches_[src][i].end(); ++it)  // for each match for this line segment
            {
                max_score = fmax(max_score,(*it).score3D_);
            }
        }

        // scores must be at least a certain percentage of the best
        float score_lim = L3D_DEF_MIN_BEST_SCORE_PERC*max_score;

        std::cout << prefix_ << "scoring: max_score = " << max_score << std::endl;

        unsigned int num_valid = 0;  // num_valid this source views
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<matches_[src].size(); ++i)
        {
            L3DPP::Match best_match;    // best match for a line segment
            best_match.score3D_ = 0.0f;

            std::list<L3DPP::Match> matches = matches_[src][i];
            matches_[src][i].clear();
            std::list<L3DPP::Match>::const_iterator it = matches.begin();
            for(; it!=matches.end(); ++it)
            {
                if((*it).score3D_ > 0.0f && (*it).score3D_ > score_lim)
                {
                    matches_[src][i].push_back(*it);

                    if((*it).score3D_ > best_match.score3D_)
                        best_match = (*it);
                }
            }

            scoring_mutex_.lock();
            num_valid += matches_[src][i].size();
            scoring_mutex_.unlock();

            // store best match as estimated 3D position
            if(best_match.score3D_ > L3D_DEF_MIN_BEST_SCORE_3D)
            {
                L3DPP::Segment2D seg(src,i);
                // use src or tgt to estimate the 3D position? true or false?
                L3DPP::Segment3D seg3D = unprojectMatch(best_match,true);
                best_match_mutex_.lock();
                // record the ID of the estimated_position3D and the cameraID and line segment ID
                entry_map_[seg] = estimated_position3D_.size();
                estimated_position3D_.push_back(std::pair<L3DPP::Segment3D,L3DPP::Match>(seg3D,best_match));

                // store depths
                depths.push_back(best_match.depth_p1_);
                depths.push_back(best_match.depth_p2_);
                best_match_mutex_.unlock();
            }
            else
            {
                // remove matches for this line segment
                matches_[src][i].clear();
            }
        }

        num_matches_[src] = num_valid;
        // median depth for this view
        float med_depth = L3D_EPS;
        if(depths.size() > 0)
        {
            std::sort(depths.begin(),depths.end());
            med_depth = depths[depths.size()/2];
        }

        if(!fixed3Dregularizer_)
            views_[src]->update_median_depth(med_depth,-1.0f,med_scene_depth_);
        else
            views_[src]->update_median_depth(med_depth,sigma_p_,med_scene_depth_);
    }

    //------------------------------------------------------------------------------
    void Line3D::storeInverseMatches(const unsigned int src)
    {
        for(size_t i=0; i<matches_[src].size(); ++i)
        {
            std::list<L3DPP::Match>::const_iterator it = matches_[src][i].begin();
            for(; it!=matches_[src][i].end(); ++it)
            {
                L3DPP::Match m = *it;
                if(m.score3D_ > 0.0f && !processed_[m.tgt_camID_])
                {
                    L3DPP::Match m_inv;
                    m_inv = m;
                    m_inv.src_camID_ = m.tgt_camID_;
                    m_inv.src_segID_ = m.tgt_segID_;
                    m_inv.tgt_camID_ = m.src_camID_;
                    m_inv.tgt_segID_ = m.src_segID_;
                    m_inv.depth_p1_ = m.depth_q1_;
                    m_inv.depth_p2_ = m.depth_q2_;
                    m_inv.depth_q1_ = m.depth_p1_;
                    m_inv.depth_q2_ = m.depth_p2_;
                    m_inv.score3D_ = 0.0f;          //TODO::why here not m_inv.score3D = m.score3D?
                    // well, cause m_inv.score3D != m.score3D, it should be compute by scoreCPU, here is a initialization

                    matches_[m.tgt_camID_][m.tgt_segID_].push_back(m_inv);
                    ++num_matches_[m.tgt_camID_];
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::reconstruct3Dlines(const unsigned int visibility_t, const bool perform_diffusion,
                                    const float collinearity_t, const bool use_G2O,
                                    const unsigned int max_iter_G2O)
    {
        // no views can be added during reconstruction!
        view_mutex_.lock();
        view_reserve_mutex_.lock();

        std::cout <<std::endl << prefix_ << "[3] RECONSTRUCTION ===============================" << std::endl;

        if(estimated_position3D_.size() == 0)
        {
            std::cout << prefix_wng_ << "no clusterable segments! forgot to match lines?" << std::endl;
            view_reserve_mutex_.unlock();
            view_mutex_.unlock();
            return;
        }

        // init
        max_iter_G2O_ = max_iter_G2O;
        visibility_t_ = std::max(int(visibility_t),3);
        clusters3D_.clear();
        lines3D_.clear();
        float prev_collin_t = collinearity_t_;
        collinearity_t_ = collinearity_t;

#ifdef L3DPP_CUDA
        perform_RDD_ = (perform_diffusion && useGPU_);
        if(perform_diffusion && !useGPU_)
            std::cout << prefix_err_ << "diffusion only possible when GPU mode enabled! using graph clustering instead..." << std::endl;
#else
        perform_RDD_ = false;
        if(perform_diffusion)
            std::cout << prefix_err_ << "diffusion not possible without CUDA! using graph clustering instead..." << std::endl;
#endif //L3DPP_CUDA

#ifdef L3DPP_G2O
        use_G2O_ = use_G2O;
#else
        use_G2O_ = false;
        if(use_G2O)
            std::cout << prefix_err_ << "CERES was not found! no optimization will be performed..." << std::endl;
#endif

        std::cout << prefix_ << "reconstructing 3D lines... [diffusion=" << perform_RDD_ << ", use g2o =" << use_G2O_ << "]" << std::endl;

        // translate, for all views, fix their center of gravity as original of coordinates
        translate();

        // find collinear segments (if not already done)
        if(collinearity_t_ > L3D_EPS && (prev_collin_t < L3D_EPS || fabs(prev_collin_t-collinearity_t_) > L3D_EPS))
        {
            std::cout << prefix_ << "find collinear segments... [" << collinearity_t_ <<" px]" << std::endl;
            findCollinearSegments();
        }

        // compute median scene depth for lines
        std::vector<float> scene_depths_lines;
        for(std::map<unsigned int,L3DPP::View*>::const_iterator vit=views_.begin(); vit!=views_.end(); ++vit)
        {
            if(vit->second->median_depth() > L3D_EPS)
                scene_depths_lines.push_back(vit->second->median_depth());
        }

        if(scene_depths_lines.size() > 0)
        {
            std::sort(scene_depths_lines.begin(),scene_depths_lines.end());
            med_scene_depth_lines_ = scene_depths_lines[scene_depths_lines.size()/2];
        }
        else
        {
            med_scene_depth_lines_ = 0.0f;
        }

        // compute affinity matrix, compute the structure A_ and assign every interested segment2D and ID
        std::cout << prefix_ << "computing affinity matrix..." << std::endl;
        computingAffinityMatrix();

        std::cout << prefix_ << "A: ";
        std::cout << "#entries=" << A_.size() << ", #rows=" << global2local_.size();

        unsigned int perc = float(global2local_.size())/float(num_lines_total_)*100.0f;
        std::cout << " [~" << perc << "%]" << std::endl;

        // perform diffusion
        if(perform_RDD_)
        {
            std::cout << prefix_ << "matrix diffusion..." << std::endl;
            performRDD();
        }

        // cluster matrix
        std::cout << prefix_ << "clustering segments..." << std::endl;
        clusterSegments();  // compute cluster3D_

        global2local_.clear();
        local2global_.clear();

        // optimize
        if(use_G2O_)
        {
            std::cout << prefix_ << "optimizing 3D lines..." << std::endl;
            optimizeClusters();
        }

        // compute final 3D segments
        std::cout << prefix_ << "computing final 3D lines..." << std::endl;
        computeFinal3Dsegments();

        clusters3D_.clear();

        // filter tiny (noisy) segments
        std::cout << prefix_ << "filtering tiny segments..." << std::endl;
        filterTinySegments();

        std::cout << prefix_ << "3D lines: total=" << lines3D_.size() << std::endl;

        // untranslate
        untranslate();

        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::findCollinearSegments()
    {
        if(collinearity_t_ > L3D_EPS)
        {
            std::map<unsigned int,L3DPP::View*>::iterator it=views_.begin();
            unsigned int i=0;
            for(; it!=views_.end(); ++it,++i)
            {
                it->second->findCollinearSegments(collinearity_t_,useGPU_);

                if(i%10 == 0)
                {
                    if(i != 0)
                        std::cout << std::endl;

                    std::cout << prefix_;
                }

                std::cout << "[" << std::setfill('0') << std::setw(L3D_DISP_CAMS) << it->first << "] ";
            }
            std::cout << std::endl;
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::computingAffinityMatrix()
    {
        // reset
        A_.clear();
        global2local_.clear();
        local2global_.clear();
        localID_ = 0;
        used_.clear();

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<estimated_position3D_.size(); ++i)
        {
            L3DPP::Segment3D seg3D = estimated_position3D_[i].first;
            L3DPP::Match m = estimated_position3D_[i].second;        // best_match for one line segment
            L3DPP::Segment2D seg2D(m.src_camID_,m.src_segID_);
            bool found_aff = false;
            int id1 = -1;

            // iterate over matches
            // now these matches are filtered, only reliable matches remained
            std::list<L3DPP::Match>::const_iterator m_it = matches_[m.src_camID_][m.src_segID_].begin();
            for(; m_it!=matches_[m.src_camID_][m.src_segID_].end(); ++m_it)
            {
                L3DPP::Match m2 = *m_it;
                L3DPP::Segment2D seg2D2(m2.tgt_camID_,m2.tgt_segID_);

                // this similarity is not used for judge which match is reasonable, but to find similarity
                // between those "best matches" on 2D plane
                float sim = similarity(seg3D,m,seg2D2,false);

                if(sim > L3D_DEF_MIN_AFFINITY && unused(seg2D,seg2D2))
                {
                    // check IDs
                    if(id1 < 0)
                        id1 = getLocalID(seg2D); // here, all segment2D is assigned an ID successively, start from 0

                    int id2 = getLocalID(seg2D2);

                    // push into affinity matrix
                    aff_mat_mutex_.lock();

                    CLEdge e;
                    e.i_ = id1;
                    e.j_ = id2;
                    e.w_ = sim;
                    A_.push_back(e);
                    e.i_ = id2;
                    e.j_ = id1;
                    A_.push_back(e);
                    found_aff = true;

                    aff_mat_mutex_.unlock();

                    // add links to potentially collinear segments to tgt
                    if(collinearity_t_ > L3D_EPS)
                    {
                        L3DPP::View* v = views_[seg2D2.camID()];
                        std::list<unsigned int> coll = v->collinearSegments(seg2D2.segID());

                        std::list<unsigned int>::const_iterator cit = coll.begin();
                        for(; cit!=coll.end(); ++cit)
                        {
                            L3DPP::Segment2D seg2D2_coll(seg2D2.camID(),*cit);

                            float sim = similarity(seg3D,m,seg2D2_coll,false);

                            if(sim > L3D_DEF_MIN_AFFINITY && unused(seg2D,seg2D2_coll))
                            {
                                // check IDs
                                int id2 = getLocalID(seg2D2_coll);

                                // push into affinity matrix
                                aff_mat_mutex_.lock();

                                CLEdge e;
                                e.i_ = id1;
                                e.j_ = id2;
                                e.w_ = sim;
                                A_.push_back(e);
                                e.i_ = id2;
                                e.j_ = id1;
                                A_.push_back(e);

                                aff_mat_mutex_.unlock();
                            }
                        }
                    }
                }
            }

            // add links to potentially collinear segments
            if(found_aff && id1 >= 0 && collinearity_t_ > L3D_EPS)
            {
                L3DPP::View* v = views_[seg2D.camID()];
                std::list<unsigned int> coll = v->collinearSegments(seg2D.segID());

                std::list<unsigned int>::const_iterator cit = coll.begin();
                for(; cit!=coll.end(); ++cit)
                {
                    L3DPP::Segment2D seg2D_coll(seg2D.camID(),*cit);

                    float sim = similarity(seg3D,m,seg2D_coll,false);

                    if(sim > L3D_DEF_MIN_AFFINITY && unused(seg2D,seg2D_coll))
                    {
                        // check IDs
                        int id2 = getLocalID(seg2D_coll);

                        // push into affinity matrix
                        aff_mat_mutex_.lock();

                        CLEdge e;
                        e.i_ = id1;
                        e.j_ = id2;
                        e.w_ = sim;
                        A_.push_back(e);
                        e.i_ = id2;
                        e.j_ = id1;
                        A_.push_back(e);

                        aff_mat_mutex_.unlock();
                    }
                }
            }
        }

        // cleanup
        used_.clear();
    }

    //------------------------------------------------------------------------------
    bool Line3D::unused(const Segment2D &seg1, const Segment2D &seg2)
    {
        bool unused = true;

        // check if used
        aff_used_mutex_.lock();
        if(used_[seg1].find(seg2) != used_[seg1].end())
        {
            // already used
            unused = false;
        }
        else
        {
            // not yet used
            used_[seg1].insert(seg2);
            used_[seg2].insert(seg1);
        }
        aff_used_mutex_.unlock();

        return unused;
    }

    //------------------------------------------------------------------------------
    int Line3D::getLocalID(const Segment2D &seg)
    {
        int id;
        aff_id_mutex_.lock();
        if(global2local_.find(seg) == global2local_.end()) // if not processed, push_back it and give it an id = size+1
        {
            id = localID_;  // start from 0
            ++localID_;

            global2local_[seg] = id;
            local2global_[id] = seg;
        }
        else // if processed and been stored, just return the ID
        {
            id = global2local_[seg];
        }
        aff_id_mutex_.unlock();
        return id;
    }

    //------------------------------------------------------------------------------
    void Line3D::performRDD()
    {
#ifdef L3DPP_CUDA
        // create sparse GPU matrix
        L3DPP::SparseMatrix* W = new L3DPP::SparseMatrix(A_,global2local_.size());

        // perform RDD
        L3DPP::replicator_dynamics_diffusion_GPU(W,prefix_);

        // update affinities (symmetrify)
        W->download();
        A_.clear();

        std::map<int,std::map<int,float> > entries;
        for(unsigned int i=0; i<W->entries()->width(); ++i)
        {
            int s1 = W->entries()->dataCPU(i,0)[0].x;
            int s2 = W->entries()->dataCPU(i,0)[0].y;
            float w12 = W->entries()->dataCPU(i,0)[0].z;

            float w21 = w12;
            if(entries[s2].find(s1) != entries[s2].end())
            {
                // other one already processed
                w21 = entries[s2][s1];
            }

            float w = fmin(w12,w21);

            entries[s1][s2] = w;
            entries[s2][s1] = w;
        }

        std::map<int,std::map<int,float> >::const_iterator it = entries.begin();
        for(; it!=entries.end(); ++it)
        {
            std::map<int,float>::const_iterator it2 = it->second.begin();
            for(; it2!=it->second.end(); ++it2)
            {
                CLEdge e;
                e.i_ = it->first;
                e.j_ = it2->first;
                e.w_ = it2->second;
                A_.push_back(e);
            }
        }

        // cleanup
        delete W;
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    void Line3D::clusterSegments()
    {
        // init
        clusters3D_.clear();
        lines3D_.clear();

        if(A_.size() == 0)
            return;

        // graph clustering, return a joined graph of all elements in A_
        L3DPP::CLUniverse* u = L3DPP::performClustering(A_,global2local_.size(),3.0f);

        // clustering done
        A_.clear();

        //process clusters
        // store all segment2D for a cluster, [cluster ID, Segment2Ds]
        std::map<int,std::list<L3DPP::Segment2D> > cluster2segments;
        // store all image IDs for a cluster, [cluser ID, image IDs(all value is true)]
        std::map<int,std::map<unsigned int,bool> > cluster2cameras;
        // store all unique cluster IDs, like 3, 4, 7, 9, ..., in this case, maybe 0,1,2 all connected to 3,
        // so they are discard, i.e., will not appear in unique_clusters
        std::vector<int> unique_clusters;

        std::map<int,L3DPP::Segment2D>::const_iterator it = local2global_.begin();
        for(; it!=local2global_.end(); ++it)
        {
            int clID = u->find(it->first);    // return the clusterID
            L3DPP::Segment2D seg = it->second;

            if(cluster2segments.find(clID) == cluster2segments.end())
                unique_clusters.push_back(clID);

            // store segment
            cluster2segments[clID].push_back(seg);
            // store camera
            cluster2cameras[clID][seg.camID()] = true;
        }
        delete u;

        if(cluster2segments.size() == 0)
        {
            std::cout << prefix_wng_ << "no clusters found..." << std::endl;
            return;
        }

        std::cout << prefix_ << "clusters: ";
        std::cout << "total=" << cluster2segments.size() << ", ";

        // create 3D lines for valid clusters
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<unique_clusters.size(); ++i)
        {
            int clID = unique_clusters[i];

            if(cluster2cameras[clID].size() >= visibility_t_)  // this cluster must have 3 images
            {
                // create 3D line cluster
                L3DPP::LineCluster3D LC = get3DlineFromCluster(cluster2segments[clID]);

                if(LC.size() > 0)
                {
                    // 3D line valid --> store in list
                    cluster_mutex_.lock();
                    clusters3D_.push_back(LC);
                    cluster_mutex_.unlock();
                }
            }
        }
        std::cout << "valid=" << clusters3D_.size();

        unsigned int perc = float(clusters3D_.size())/float(cluster2segments.size())*100;
        std::cout << " [~" << perc << "%]";

        std::cout << std::endl;
    }

    //------------------------------------------------------------------------------
    L3DPP::LineCluster3D Line3D::get3DlineFromCluster(const std::list<L3DPP::Segment2D>& cluster)
    {
        // create scatter matrix
        Eigen::Vector3d P(0,0,0);
        int n = cluster.size()*2;
        Eigen::MatrixXd L_points(3,n);

        std::list<L3DPP::Segment2D>::const_iterator it = cluster.begin();
        // the image ID of this cluster, i.e., the segment2D in this cluster who is the longest
        unsigned int reference_cam = 0;
        float max_len_2D = 0.0f;
        for(size_t i=0; it!=cluster.end(); ++it,i+=2)
        {
            // get 3D hypothesis
            size_t pos = entry_map_[*it];
            L3DPP::Segment3D hyp3D = estimated_position3D_[pos].first;

            P += hyp3D.P1();
            P += hyp3D.P2();

            L_points(0,i) = hyp3D.P1().x();
            L_points(1,i) = hyp3D.P1().y();
            L_points(2,i) = hyp3D.P1().z();

            L_points(0,i+1) = hyp3D.P2().x();
            L_points(1,i+1) = hyp3D.P2().y();
            L_points(2,i+1) = hyp3D.P2().z();

            // check 2D length -> max length defines reference view (for filtering later on)
            Eigen::Vector4f coords = views_[(*it).camID()]->getLineSegment2D((*it).segID());
            float length_sqr = (coords(0)-coords(2))*(coords(0)-coords(2)) + (coords(1)-coords(3))*(coords(1)-coords(3));
            if(length_sqr > max_len_2D)
            {
                max_len_2D = length_sqr;
                reference_cam = (*it).camID();
            }
        }

        // center of gravity
        P /= double(n);

        // direction
        Eigen::MatrixXd C = Eigen::MatrixXd::Identity(n,n)-(1.0/(double)(n))*Eigen::MatrixXd::Constant(n,n,1.0);
        Eigen::MatrixXd Scat = L_points*C*L_points.transpose();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Scat, Eigen::ComputeThinU);

        Eigen::MatrixXd U;
        Eigen::VectorXd S;

        U = svd.matrixU();
        S = svd.singularValues();

        int maxPos;
        S.maxCoeff(&maxPos);

        Eigen::Vector3d dir = Eigen::Vector3d(U(0, maxPos), U(1, maxPos), U(2, maxPos));
        dir.normalize();

        // initial 3D line for cluster
        L3DPP::Segment3D initial3Dline(P-dir,P+dir);  // the computed identity segment3D, center and direction
        L3DPP::LineCluster3D LC = L3DPP::LineCluster3D(initial3Dline,cluster,reference_cam);

        return LC;
    }

    //------------------------------------------------------------------------------
    L3DPP::Segment3D Line3D::project2DsegmentOnto3Dline(const L3DPP::Segment2D& seg2D,
                                                        const L3DPP::Segment3D& seg3D,
                                                        bool& success)
    {
        // tgt line
        Eigen::Vector3d P = seg3D.P1();
        Eigen::Vector3d u = seg3D.dir();

        // src line 1
        L3DPP::View* v = views_[seg2D.camID()];
        Eigen::Vector3d Q = v->C();
        Eigen::Vector3d v1 = v->getNormalizedLinePointRay(seg2D.segID(),true);

        // src line 2
        Eigen::Vector3d v2 = v->getNormalizedLinePointRay(seg2D.segID(),false);

        Eigen::Vector3d w = P-Q;

        // vals
        double a = u.dot(u);
        double b1 = u.dot(v1);
        double b2 = u.dot(v2);
        double c1 = v1.dot(v1);
        double c2 = v2.dot(v2);
        double d = u.dot(w);
        double e1 = v1.dot(w);
        double e2 = v2.dot(w);

        double denom1 = a*c1 - b1*b1;
        double denom2 = a*c2 - b2*b2;

        if(fabs(denom1) > L3D_EPS && fabs(denom2) > L3D_EPS)
        {
            success = true;
            double s1 = (b1*e1 - c1*d)/denom1;
            double s2 = (b2*e2 - c2*d)/denom2;

            return L3DPP::Segment3D(P+s1*u,P+s2*u);
        }
        else
        {
            // projection not possible, exactly on the epipolar lines
            success = false;
            return L3DPP::Segment3D();
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::optimizeClusters()
    {
#ifdef L3DPP_G2O
        L3DPP::LineOptimizer opt(views_,&clusters3D_,max_iter_G2O_,prefix_);
        opt.optimize();
#endif //L3DPP_CERES
    }

    //------------------------------------------------------------------------------
    void Line3D::computeFinal3Dsegments()
    {
        // iterate over clusters and find all valid collinear segments
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<clusters3D_.size(); ++i)
        {
            std::list<L3DPP::Segment3D> collinear = findCollinearSegments(clusters3D_[i]);

            if(collinear.size() > 0)
            {
                L3DPP::FinalLine3D final;
                final.collinear3Dsegments_ = collinear;
                final.underlyingCluster_ = clusters3D_[i];

                cluster_mutex_.lock();
                lines3D_.push_back(final);
                cluster_mutex_.unlock();
            }
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::filterTinySegments()
    {
        // remove 3D segments that are too small
        size_t valid_before = lines3D_.size();
        if(valid_before == 0)
            return;

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(size_t i=0; i<lines3D_.size(); ++i)
        {
            L3DPP::View* v = views_[lines3D_[i].underlyingCluster_.reference_view()];

            std::list<L3DPP::Segment3D> filteredSegments;
            std::list<L3DPP::Segment3D>::const_iterator it = lines3D_[i].collinear3Dsegments_.begin();
            for(; it!=lines3D_[i].collinear3Dsegments_.end(); ++it)
            {
                if(v->projectedLongEnough(*it))
                    filteredSegments.push_back(*it);
            }

            lines3D_[i].collinear3Dsegments_ = filteredSegments;
        }

        // remove invalid lines
        std::vector<L3DPP::FinalLine3D> lines3D;
        for(size_t i=0; i<lines3D_.size(); ++i)
        {
            if(lines3D_[i].collinear3Dsegments_.size() > 0)
                lines3D.push_back(lines3D_[i]);
        }
        lines3D_ = lines3D;

        size_t valid_after = lines3D_.size();

        std::cout << prefix_ << "removed lines: " << valid_before-valid_after << std::endl;
    }

    //------------------------------------------------------------------------------
    std::list<L3DPP::Segment3D> Line3D::findCollinearSegments(const L3DPP::LineCluster3D& cluster)
    {
        // project onto 3D line
        std::list<L3DPP::Segment3D> collinear_segments;
        Eigen::Vector3d COG = 0.5*(cluster.seg3D().P1()+cluster.seg3D().P2());
        std::list<L3DPP::Segment2D>::const_iterator it = cluster.residuals()->begin();

        std::list<L3DPP::PointOn3DLine> linePoints;
        std::vector<Eigen::Vector3d> pts(cluster.residuals()->size()*2);

        float distToCOG = 0.0f;
        Eigen::Vector3d border;

        size_t pID = 0;
        for(size_t id=0; it!=cluster.residuals()->end(); ++it,++id,pID+=2)
        {
            // project onto 3D line
            bool success;
            L3DPP::Segment3D proj = project2DsegmentOnto3Dline(*it,cluster.seg3D(),success);

            if(success)
            {
                // create line points
                L3DPP::PointOn3DLine p1,p2;

                p1.lineID_ = id;
                p1.pointID_ = pID;
                p1.camID_ = (*it).camID();
                pts[pID] = proj.P1();
                linePoints.push_back(p1);

                float d = (proj.P1()-COG).norm();
                if(d > distToCOG)
                {
                    distToCOG = d;
                    border = proj.P1();
                }

                p2.lineID_ = id;
                p2.pointID_ = pID+1;
                p2.camID_ = (*it).camID();
                pts[pID+1] = proj.P2();
                linePoints.push_back(p2);

                d = (proj.P2()-COG).norm();
                if(d > distToCOG)
                {
                    distToCOG = d;
                    border = proj.P2();
                }
            }
        }

        // check number of projected lines/points
        if(linePoints.size() < 6)
            return collinear_segments;

        // sort by distance to border
        std::list<L3DPP::PointOn3DLine>::iterator lit = linePoints.begin();
        for(; lit!=linePoints.end(); ++lit)
        {
            (*lit).distToBorder_ = (pts[(*lit).pointID_]-border).norm();
        }
        linePoints.sort(L3DPP::sortPointsOn3DLine);

        // iterate and create 3D segments
        std::map<size_t,unsigned int> open;
        std::map<size_t,bool> open_lines;
        bool opened = false;
        Eigen::Vector3d current_start(0,0,0);
        lit = linePoints.begin();
        for(; lit!=linePoints.end(); ++lit)
        {
            L3DPP::PointOn3DLine pt = *lit;

            if(open_lines.find(pt.lineID_) == open_lines.end())
            {
                // opening
                open_lines[pt.lineID_] = true;

                if(open.find(pt.camID_) == open.end())
                    open[pt.camID_] = 1;
                else
                    ++open[pt.camID_];
            }
            else
            {
                // closing
                open_lines.erase(pt.lineID_);

                --open[pt.camID_];

                if(open[pt.camID_] == 0)
                    open.erase(pt.camID_);
            }

            if(opened && open.size() < 3)
            {
                L3DPP::Segment3D l(current_start,pts[pt.pointID_]);
                collinear_segments.push_back(l);
                opened = false;
            }
            else if(!opened && open.size() >= 3)
            {
                current_start = pts[pt.pointID_];
                opened = true;
            }
        }

        return collinear_segments;
    }

    //------------------------------------------------------------------------------
    void Line3D::get3Dlines(std::vector<L3DPP::FinalLine3D>& result)
    {
        view_mutex_.lock();
        view_reserve_mutex_.lock();
        result = lines3D_;
        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::saveResultAsSTL(const std::string& output_folder)
    {
        view_mutex_.lock();
        view_reserve_mutex_.lock();

        if(lines3D_.size() == 0)
        {
            std::cout << prefix_wng_ << "no 3D lines to save!" << std::endl;
            view_reserve_mutex_.unlock();
            view_mutex_.unlock();
            return;
        }

        // get filename
        std::string filename = output_folder+"/"+createOutputFilename()+".stl";

        std::ofstream file, fout;
        file.open(filename.c_str());
        fout.open(output_folder+"/"+"lightlines.txt");

        file << "solid lineModel" << std::endl;

        for(size_t i=0; i<lines3D_.size(); ++i)
        {
            L3DPP::FinalLine3D current = lines3D_[i];

            std::list<L3DPP::Segment3D>::const_iterator it2 = current.collinear3Dsegments_.begin();
            for(; it2!=current.collinear3Dsegments_.end(); ++it2)
            {
                Eigen::Vector3d P1 = (*it2).P1();
                Eigen::Vector3d P2 = (*it2).P2();

                char x1[50];
                char y1[50];
                char z1[50];

                char x2[50];
                char y2[50];
                char z2[50];

                sprintf(x1,"%e",P1.x());
                sprintf(y1,"%e",P1.y());
                sprintf(z1,"%e",P1.z());

                sprintf(x2,"%e",P2.x());
                sprintf(y2,"%e",P2.y());
                sprintf(z2,"%e",P2.z());

                file << " facet normal 1.0e+000 0.0e+000 0.0e+000" << std::endl;
                file << "  outer loop" << std::endl;
                file << "   vertex " << x1 << " " << y1 << " " << z1 << std::endl;
                file << "   vertex " << x2 << " " << y2 << " " << z2 << std::endl;
                file << "   vertex " << x1 << " " << y1 << " " << z1 << std::endl;
                file << "  endloop" << std::endl;
                file << " endfacet" << std::endl;

                fout << P1.x() << " " << P1.y() << " " << P1.z() << " "
                     << P2.x() << " " << P2.y() << " " << P2.z() << std::endl;
            }
        }

        file << "endsolid lineModel" << std::endl;
        file.close();

        fout.close();

        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::saveTempResultAsSTL(const std::string& output_folder,
                                     const std::string& suffix,
                                     const std::vector<L3DPP::Segment3D>& lines3D)
    {
        // get filename
        std::string filename = output_folder+"/"+createOutputFilename()+"__"+suffix+".stl";

        std::ofstream file;
        file.open(filename.c_str());

        file << "solid lineModel" << std::endl;

        for(size_t i=0; i<lines3D.size(); ++i)
        {
            L3DPP::Segment3D current = lines3D[i];

            Eigen::Vector3d P1 = current.P1();
            Eigen::Vector3d P2 = current.P2();

            char x1[50];
            char y1[50];
            char z1[50];

            char x2[50];
            char y2[50];
            char z2[50];

            sprintf(x1,"%e",P1.x());
            sprintf(y1,"%e",P1.y());
            sprintf(z1,"%e",P1.z());

            sprintf(x2,"%e",P2.x());
            sprintf(y2,"%e",P2.y());
            sprintf(z2,"%e",P2.z());

            file << " facet normal 1.0e+000 0.0e+000 0.0e+000" << std::endl;
            file << "  outer loop" << std::endl;
            file << "   vertex " << x1 << " " << y1 << " " << z1 << std::endl;
            file << "   vertex " << x2 << " " << y2 << " " << z2 << std::endl;
            file << "   vertex " << x1 << " " << y1 << " " << z1 << std::endl;
            file << "  endloop" << std::endl;
            file << " endfacet" << std::endl;
        }

        file << "endsolid lineModel" << std::endl;
        file.close();
    }

    //------------------------------------------------------------------------------
    void Line3D::saveResultAsOBJ(const std::string& output_folder)
    {
        view_mutex_.lock();
        view_reserve_mutex_.lock();

        if(lines3D_.size() == 0)
        {
            std::cout << prefix_wng_ << "no 3D lines to save!" << std::endl;
            view_reserve_mutex_.unlock();
            view_mutex_.unlock();
            return;
        }

        // get filename
        std::string filename = output_folder+"/"+createOutputFilename()+".obj";

        std::ofstream file;
        file.open(filename.c_str());

        size_t lineID = 0;
        size_t pointID = 1;
        std::map<size_t,size_t> lines2points;
        for(size_t i=0; i<lines3D_.size(); ++i)
        {
            L3DPP::FinalLine3D current = lines3D_[i];

            std::list<L3DPP::Segment3D>::const_iterator it2 = current.collinear3Dsegments_.begin();
            for(; it2!=current.collinear3Dsegments_.end(); ++it2,++lineID,pointID+=2)
            {
                Eigen::Vector3d P1 = (*it2).P1();
                Eigen::Vector3d P2 = (*it2).P2();

                file << "v " << P1.x() << " " << P1.y() << " " << P1.z() << std::endl;
                file << "v " << P2.x() << " " << P2.y() << " " << P2.z() << std::endl;

                lines2points[lineID] = pointID;
            }
        }

        std::map<size_t,size_t>::const_iterator it = lines2points.begin();
        for(; it!=lines2points.end(); ++it)
        {
            file << "l " << it->second << " " << it->second+1 << std::endl;
        }

        file.close();

        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::save3DLinesAsTXT(const std::string& output_folder)
    {
        view_mutex_.lock();
        view_reserve_mutex_.lock();

        if(lines3D_.size() == 0)
        {
            std::cout << prefix_wng_ << "no 3D lines to save!" << std::endl;
            view_reserve_mutex_.unlock();
            view_mutex_.unlock();
            return;
        }

        // get filename
        std::string filename = output_folder+"/"+createOutputFilename()+".txt";

        std::ofstream file;
        file.open(filename.c_str());

        for(size_t i=0; i<lines3D_.size(); ++i)
        {
            L3DPP::FinalLine3D current = lines3D_[i];

            if(current.collinear3Dsegments_.size() == 0)
                continue;

            // write 3D segments
            file << current.collinear3Dsegments_.size() << " ";
            std::list<L3DPP::Segment3D>::const_iterator it2 = current.collinear3Dsegments_.begin();
            for(; it2!=current.collinear3Dsegments_.end(); ++it2)
            {
                Eigen::Vector3d P1 = (*it2).P1();
                Eigen::Vector3d P2 = (*it2).P2();

                file << P1.x() << " " << P1.y() << " " << P1.z() << " ";
                file << P2.x() << " " << P2.y() << " " << P2.z() << " ";
            }

            // write 2D residuals
            file << current.underlyingCluster_.residuals()->size() << " ";
            std::list<L3DPP::Segment2D>::const_iterator it3 = current.underlyingCluster_.residuals()->begin();
            for(; it3!=current.underlyingCluster_.residuals()->end(); ++it3)
            {
                file << (*it3).camID() << " " << (*it3).segID() << " ";
                Eigen::Vector4f coords = getSegmentCoords2D(*it3);
                file << coords(0) << " " << coords(1) << " ";
                file << coords(2) << " " << coords(3) << " ";
            }

            file << std::endl;
        }

        file.close();

        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::save3DLinesAsBIN(const std::string& output_folder)
    {
        view_mutex_.lock();
        view_reserve_mutex_.lock();

        if(lines3D_.size() == 0)
        {
            std::cout << prefix_wng_ << "no 3D lines to save!" << std::endl;
            view_reserve_mutex_.unlock();
            view_mutex_.unlock();
            return;
        }

        // get filename
        std::string filename = output_folder+"/"+createOutputFilename()+".bin";

        // serialize
        L3DPP::serializeToFile(filename,lines3D_);

        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    Eigen::Matrix3d Line3D::rotationFromRPY(const double roll, const double pitch,
                                            const double yaw)
    {
        const Eigen::Matrix3d Rx
          = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
        const Eigen::Matrix3d Ry
          = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
        const Eigen::Matrix3d Rz
          = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

        const Eigen::Matrix3d R = Rz * Ry * Rx;

        return R;
    }

    //------------------------------------------------------------------------------
    Eigen::Matrix3d Line3D::rotationFromQ(const double Qw, const double Qx,
                                          const double Qy, const double Qz)
    {
        double n = Qw*Qw + Qx*Qx + Qy*Qy + Qz*Qz;

        double s;
        if(fabs(n) < L3D_EPS)
        {
            s = 0;
        }
        else
        {
            s = 2.0/n;
        }

        double wx = s*Qw*Qx; double wy = s*Qw*Qy; double wz = s*Qw*Qz;
        double xx = s*Qx*Qx; double xy = s*Qx*Qy; double xz = s*Qx*Qz;
        double yy = s*Qy*Qy; double yz = s*Qy*Qz; double zz = s*Qz*Qz;

        Eigen::Matrix3d R;
        R(0,0) = 1.0 - (yy + zz); R(0,1) = xy - wz;         R(0,2) = xz + wy;
        R(1,0) = xy + wz;         R(1,1) = 1.0 - (xx + zz); R(1,2) = yz - wx;
        R(2,0) = xz - wy;         R(2,1) = yz + wx;         R(2,2) = 1.0 - (xx + yy);
        return R;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector4f Line3D::getSegmentCoords2D(const L3DPP::Segment2D& seg2D)
    {
        Eigen::Vector4f coords(0,0,0,0);
        if(views_.find(seg2D.camID()) != views_.end())
        {
            coords = views_[seg2D.camID()]->getLineSegment2D(seg2D.segID());
        }
        return coords;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector4f Line3D::getSegmentCoords2D(const unsigned int camID,
                                               const unsigned int segID)
    {
        return getSegmentCoords2D(L3DPP::Segment2D(camID,segID));
    }

    //------------------------------------------------------------------------------
    void Line3D::eigen2dataArray(L3DPP::DataArray<float>* &DA, const Eigen::MatrixXd& M)
    {
        DA = new L3DPP::DataArray<float>(M.cols(),M.rows());
        for(size_t y=0; y<size_t(M.rows()); ++y)
            for(size_t x=0; x<size_t(M.cols()); ++x)
                DA->dataCPU(x,y)[0] = M(y,x);
    }

    //------------------------------------------------------------------------------
    void Line3D::decomposeProjectionMatrix(const Eigen::MatrixXd P_in,
                                           Eigen::Matrix3d& K_out,
                                           Eigen::Matrix3d& R_out,
                                           Eigen::Vector3d& t_out)
    {
        if(P_in.rows() != 3 && P_in.cols() != 4)
        {
            std::cout << "P is not a 3x4 matrix! (" << P_in.rows() << "x" << P_in.cols() << ")" << std::endl;
            return;
        }

        K_out = P_in.block<3,3>(0,0);

        // get affine matrix (rq-decomposition of M)
        // See Hartley & Zissermann, p552 (1st ed.)
        double h = std::sqrt((K_out(2,1))*(K_out(2,1)) + (K_out(2,2))*(K_out(2,2)));
        double s =  K_out(2,1) / h;
        double c = -K_out(2,2) / h;

        Eigen::Matrix3d Rx;
        Rx.setZero();
        Rx(0,0) =  1;
        Rx(1,1) =  c; Rx(2,2) = c;
        Rx(1,2) = -s; Rx(2,1) = s;

        K_out = K_out * Rx;

        h = sqrt((K_out(2,0))*(K_out(2,0)) + (K_out(2,2))*(K_out(2,2)));
        s =  K_out(2,0) / h;
        c = -K_out(2,2) / h;

        Eigen::Matrix3d Ry;
        Ry.setZero();
        Ry(1,1) =  1;
        Ry(0,0) =  c; Ry(2,2) = c;
        Ry(0,2) = -s; Ry(2,0) = s;

        K_out = K_out * Ry;

        h = sqrt((K_out(1,0)*K_out(1,0)) + (K_out(1,1)*K_out(1,1)));
        s =  K_out(1,0) / h;
        c = -K_out(1,1) / h;

        Eigen::Matrix3d Rz;
        Rz.setZero();
        Rz(2,2) =  1;
        Rz(0,0) =  c; Rz(1,1) = c;
        Rz(0,1) = -s; Rz(1,0) = s;

        K_out = K_out * Rz;

        Eigen::Matrix3d Sign = Eigen::Matrix3d::Identity(3,3);

        if (K_out(0,0) < 0) Sign(0,0) = -1;
        if (K_out(1,1) < 0) Sign(1,1) = -1;
        if (K_out(2,2) < 0) Sign(2,2) = -1;

        K_out = K_out * Sign; // change signum of columns

        R_out = Rx * Ry * Rz * Sign;
        R_out.transposeInPlace();

        Eigen::Vector3d P4;
        P4 = P_in.block<3,1>(0,3);

        t_out = K_out.inverse() * P4;

        K_out *= 1.0 / K_out(2,2); // normalize, such that lower-right element is 1
    }

    //------------------------------------------------------------------------------
    std::string Line3D::createOutputFilename()
    {
        std::stringstream str;
        str << "Line3D++__";

//        if(max_image_width_ > 0)
//            str << "W_" << max_image_width_ << "__";
//        else
//            str << "W_FULL__";

        str << "N_" << num_neighbors_ << "__";

        str << "sigmaP_" << sigma_p_ << "__";
        str << "sigmaA_" << sigma_a_ << "__";

        str << "epiOverlap_" << epipolar_overlap_ << "__";

        if(kNN_ > 0)
            str << "kNN_" << kNN_ << "__";

        if(collinearity_t_ > L3D_EPS)
            str << "COLLIN_" << collinearity_t_ << "__";

        if(fixed3Dregularizer_)
        {
            str << "FXD_SIGMA_P__";

            if(const_regularization_depth_ > 0.0f)
                str << "REG_DEPTH_" << const_regularization_depth_ << "__";
        }

        if(perform_RDD_)
            str << "DIFFUSION__";

        if(use_G2O_)
            str << "OPTIMIZED__";

        str << "vis_" << visibility_t_;
        return str.str();
    }

    L3DPP::DataArray<float4>* Line3D::addVirtualLines(const cv::Mat& image, cv::Point2f srcQuad[4], int num_lines)
    {
        L3DPP::DataArray<float4>* lines = NULL;
        lines = new L3DPP::DataArray<float4>(num_lines, 1);

        cv::Point2f dstQuad[4];
        dstQuad[0].x = 500; dstQuad[0].y = 500;
        dstQuad[1].x = 500; dstQuad[1].y = 750;
        dstQuad[2].x = 750; dstQuad[2].y = 500;
        dstQuad[3].x = 750; dstQuad[3].y = 750;

//        for (int k = 0; k < 4; ++k)
//        {
//            cout << srcQuad[k] << endl;
//        }

        cv::Mat H = cv::getPerspectiveTransform(srcQuad, dstQuad);

        int weight = image.cols;
        int height = image.rows;

        std::vector<cv::Point2f> interPts;

        int interval = 250/(num_lines-1);

        for (int i = 0; i < num_lines; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                interPts.push_back(cv::Point2f(500 + 250*j, 500 + interval*i));
            }

//            for (int j = 0; j < 2; ++j)
//            {
//                interPts.push_back(cv::Point2f(500 + interval*i, 500 + 250*j));
//            }
        }

        volatile int tmpSize = interPts.size();

        std::vector<cv::Point2f> interPair;
        interPair.reserve(num_lines);

        for (int i = 0; i < interPts.size(); ++i)
        {
            cv::Mat ptIn_3d  = (cv::Mat_<double>(3,1) << interPts[i].x, interPts[i].y, 1);
            cv::Mat ptIn_IPM = H.inv()*ptIn_3d;
            cv::Point2f ptOut;

            ptOut.x = static_cast<float>(ptIn_IPM.at<double>(0,0)/ptIn_IPM.at<double>(0,2));
            ptOut.y = static_cast<float>(ptIn_IPM.at<double>(0,1)/ptIn_IPM.at<double>(0,2));
            interPair.push_back(ptOut);
        }

//        interPair.push_back(srcQuad[0]);
//        interPair.push_back(srcQuad[2]);
//        interPair.push_back(srcQuad[1]);
//        interPair.push_back(srcQuad[3]);

//        interPair.push_back(srcQuad[0]);
//        interPair.push_back(srcQuad[1]);
//        interPair.push_back(srcQuad[2]);
//        interPair.push_back(srcQuad[3]);

        int count = num_lines-1;
//        num_lines *= 2;
        for(int32_t i = 0; i < num_lines*2; i += 2)
        {
            float4 tempf4;
            cv::Point tempSpts, tempEpts;
            tempSpts.x = interPair[i].x;
            tempSpts.y = interPair[i].y;

            tempEpts.x = interPair[i+1].x;
            tempEpts.y = interPair[i+1].y;

            tempf4.x = tempSpts.x;
            tempf4.y = tempSpts.y;
            tempf4.z = tempEpts.x;
            tempf4.w = tempEpts.y;

            lines->dataCPU(count, 0)[0] = tempf4;
            count--;
        }

        return lines;
    }

    void Line3D::findVisualNeighborsFromROIs(const unsigned int imgID)
    {
        if(visual_neighbors_.find(imgID) != visual_neighbors_.end())
        {
            // reset
            visual_neighbors_[imgID].clear();
            // views who can see common ROI with this image, [imageID, common 3D point numbers]
            std::map<unsigned int,unsigned int> commonROIs;

            /*
            std::list<unsigned int>::const_iterator wp_it = views2worldpoints_[imgID].begin();
            for(; wp_it!=views2worldpoints_[imgID].end(); ++wp_it)
            {
                // iterate over worldpoints
                unsigned int wpID = *wp_it;

                std::list<unsigned int>::const_iterator view_it = worldpoints2views_[wpID].begin();
                for(; view_it!=worldpoints2views_[wpID].end(); ++view_it)
                {
                    // all views are potential neighbors
                    unsigned int vID = *view_it;
                    if(vID != imgID)
                    {
                        if(commonWPs.find(vID) == commonWPs.end())
                        {
                            commonWPs[vID] = 1;
                        }
                        else
                        {
                            ++commonWPs[vID];
                        }
                    }
                }
            }

            if(commonWPs.size() == 0)
                return;

            // find visual neighbors
            std::list<L3DPP::VisualNeighbor> neighbors;
            L3DPP::View* v = views_[imgID];
            std::map<unsigned int,unsigned int>::const_iterator c_it = commonWPs.begin();
            for(; c_it!=commonWPs.end(); ++c_it)
            {
                unsigned int vID = c_it->first;
                unsigned int num_common_wps = c_it->second;

                VisualNeighbor vn;
                vn.imgID_ = vID;
                vn.score_ = 2.0f*float(num_common_wps)/float(num_worldpoints_[imgID]+num_worldpoints_[vID]);
                vn.axisAngle_ = v->opticalAxesAngle(views_[vID]);  // optical axe angle of v and vID
                vn.distance_score_ = v->distanceVisualNeighborScore(views_[vID]); // block distance of optical axe of v and vID

                // check baseline
                if(vn.axisAngle_ < 1.571f && num_common_wps > 4) // ~ PI/2
                {
                    neighbors.push_back(vn);
                }
            }

            // sort by score
            neighbors.sort(L3DPP::sortVisualNeighborsByScore);

            // reduce to best neighbors
            if(neighbors.size() > num_neighbors_)
            {
                // copy neighbors
                std::list<L3DPP::VisualNeighbor> neighbors_tmp = neighbors;

                // get max score
                float score_t = 0.80f*neighbors.front().score_;
                unsigned int num_bigger_t = 0;

                // count the number of highly similar views
                std::list<L3DPP::VisualNeighbor>::const_iterator nit = neighbors.begin();
                while(nit!=neighbors.end() && (*nit).score_ > score_t)
                {
                    ++num_bigger_t;
                    ++nit;
                }

                neighbors.resize(num_bigger_t); // list.resize(n), only reserve element0, element1, ..., elementn

                // resort based on projective_score and world_point_score
                neighbors.sort(L3DPP::sortVisualNeighborsByDistScore);

                if(neighbors.size() > num_neighbors_/2)
                    neighbors.resize(num_neighbors_/2);

                // combine, add the new gotten neighbors in the front of original neighbors
//                std::cout << "lightol, neighbors.size() is " << neighbors.size() << std::endl;
                neighbors.splice(neighbors.end(),neighbors_tmp);
//                std::cout << "lightol, neighbors.size() is " << neighbors.size() << std::endl;
//                int a = 1;
            }



            // highscore neighbors -> store in visual neighbor map
//            volatile float a = v->getSpecificSpatialReg(0.5f);
//            volatile float b = v->initial_median_depth();
            float min_baseline = v->getSpecificSpatialReg(0.5f)*v->initial_median_depth();
            min_baseline = 0.1f;
            std::set<unsigned int> used_neighbors;  // store IDs of qualified neighbors
            std::list<L3DPP::VisualNeighbor>::const_iterator nit = neighbors.begin();
            while(nit!=neighbors.end() && used_neighbors.size() < num_neighbors_)
            {
                L3DPP::VisualNeighbor vn = *nit;
                L3DPP::View* v2 = views_[vn.imgID_];

                // check baseline
                // v->baseLine(v2) is the baseline length of this two images
                if(used_neighbors.find(vn.imgID_) == used_neighbors.end() && v->baseLine(v2) > min_baseline)
                {
                    std::set<unsigned int>::const_iterator u_it = used_neighbors.begin();
                    bool valid = true;
                    for(; u_it!=used_neighbors.end() && valid; ++u_it)
                    {
                        if(!(v->baseLine(views_[*u_it]) > min_baseline))
                            valid = false;
                    }

                    if(valid)
                        used_neighbors.insert(vn.imgID_);
                }

                ++nit;
            }
            */

            // todo::temporarily used, should use common ROI rather than simple find back and front
            std::set<unsigned int> neighbors;  // store IDs of qualified neighbors

            for (int i = -3; i < 4; ++i)
            {
                if (i!=0 && views_.find(imgID+i) != views_.end())
                {
                    neighbors.insert(imgID+i);
                }
            }

            visual_neighbors_[imgID] = neighbors;
        }

    }

    std::map<unsigned int,Eigen::Vector2d> Line3D::calVanishPoint(std::map<unsigned int, Eigen::Matrix3d> img_R,
                                                                  std::map<unsigned int, Eigen::Vector3d> img_t,
                                                                  Eigen::Matrix3d K)
    {
        std::vector<unsigned int> imgIDs;

        std::map<unsigned int, Eigen::Matrix3d>::iterator iter = img_R.begin();
        for (; iter != img_R.end(); iter++)
        {
            imgIDs.push_back(iter->first);
        }

        std::map<unsigned int,Eigen::Vector2d> vps;
        Eigen::Matrix4d cT, nT, rT; // current pose and next pose and relative pose

        for (int i = 0; i < imgIDs.size(); ++i)
        {
            unsigned int imgID = imgIDs[i];
            cT.block(0,0,3,3) = img_R[imgID];
            cT.block(0,3,3,1) = img_t[imgID];
            cT.block(3,0,1,4) = Eigen::Vector4d(0,0,0,1).transpose();
            nT.block(0,0,3,3) = img_R[imgID+1];
            nT.block(0,3,3,1) = img_t[imgID+1];
            nT.block(3,0,1,4) = Eigen::Vector4d(0,0,0,1).transpose();
            // todo:: what is the relation ship between these R R R!
            rT = nT*cT.inverse();
            Eigen::Matrix3d R = rT.block(0,0,3,3);
            Eigen::Vector3d t = rT.block(0,3,3,1);

//            Eigen::Vector3d x3dc = -R.inverse()*t;  // camera frame

            Eigen::Vector3d cC = -img_R[imgID].inverse()*img_t[imgID];     // current optical center
            Eigen::Vector3d nC = -img_R[imgID+1].inverse()*img_t[imgID+1]; // next optical center
            Eigen::Vector3d diff = nC - cC;

            Eigen::Vector3d x3dc = img_R[imgID]*nC + img_t[imgID]; // camera coordinate

            // normlized frame
            Eigen::Vector3d x3dn = Eigen::Vector3d(x3dc.x()/x3dc.z(), x3dc.y()/x3dc.z(), 1);
            Eigen::Vector3d vp3d = K*x3dn;
            Eigen::Vector2d vp(vp3d.x(), vp3d.y());
            vps[imgID] = vp;
        }

        return vps;
    }

    Eigen::Matrix3d Line3D::rotationFromRPY(const Eigen::Vector3d rpy)
    {
        double roll = rpy[0];
        double pitch = rpy[1];
        double yaw = rpy[2];

        Eigen::AngleAxisd Rx = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd Ry = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd Rz = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitX());

        volatile Eigen::Matrix3d RX = Rx.toRotationMatrix();
        volatile Eigen::Matrix3d RY = Ry.toRotationMatrix();
        volatile Eigen::Matrix3d RZ = Rz.toRotationMatrix();

        return (Rx*Ry*Rz).toRotationMatrix();
    }
}
