#include "optimization.h"
#include <opencv2/calib3d/calib3d.hpp>

#ifdef L3DPP_G2O

namespace L3DPP
{
    //------------------------------------------------------------------------------
    void LineOptimizer::optimize()
    {
        if(clusters3D_->size() == 0)
            return;

        size_t num_cams = views_.size();
        double* cameras = new double[num_cams * CAM_PARAMETERS_SIZE];
        double* intrinsics = new double[num_cams * INTRINSIC_SIZE];
        std::map<unsigned int,size_t> cam_global2local;
        std::map<unsigned int,L3DPP::View*>::const_iterator it = views_.begin();

        for(size_t i=0; it!=views_.end(); ++it,++i)
        {
            // set local ID, assign each view an ID
            cam_global2local[it->first] = i;

            // camera (rotation and center)
            L3DPP::View* v = it->second;
            Eigen::Matrix3d rot = v->R();
            cv::Mat rotation = (cv::Mat_<double>(3, 3) << rot(0,0), rot(1,0), rot(2,0),
                    rot(0,1), rot(1,1), rot(2,1),
                    rot(0,2), rot(1,2), rot(2,2) );
            cv::Mat rotation_vector;
            cv::Rodrigues(rotation, rotation_vector);

            // axis angle
            double axis_angle[3];
            cameras[(i*CAM_PARAMETERS_SIZE) + 0] = -rotation_vector.at<double>(0);
            cameras[(i*CAM_PARAMETERS_SIZE) + 1] = -rotation_vector.at<double>(1);
            cameras[(i*CAM_PARAMETERS_SIZE) + 2] = -rotation_vector.at<double>(2);

            cameras[(i*CAM_PARAMETERS_SIZE) + 3] = (v->C())[0];
            cameras[(i*CAM_PARAMETERS_SIZE) + 4] = (v->C())[1];
            cameras[(i*CAM_PARAMETERS_SIZE) + 5] = (v->C())[2];

            // intrinsics -> cof(K)
            double fx = (v->K())(0,0);
            double fy = (v->K())(1,1);
            double px = (v->K())(0,2);
            double py = (v->K())(1,2);

            intrinsics[(i*INTRINSIC_SIZE + 0)] = px;
            intrinsics[(i*INTRINSIC_SIZE + 1)] = py;
            intrinsics[(i*INTRINSIC_SIZE + 2)] = fx;
            intrinsics[(i*INTRINSIC_SIZE + 3)] = fy;
        }

        size_t num_lines = clusters3D_->size();
        double* lines = new double[num_lines * LINE_SIZE];
        double* tmp_pts = new double[num_lines * 6];

        for(size_t i=0; i<num_lines; ++i)
        {
            L3DPP::LineCluster3D LC = clusters3D_->at(i);

            // convert to Plücker
            Eigen::Vector3d l = LC.seg3D().P2()-LC.seg3D().P1();
            l.normalize();
            Eigen::Vector3d m = (0.5*(LC.seg3D().P1()+LC.seg3D().P2())).cross(l);

            // convert to Cayley [Zhang and Koch, J. Vis. Commun. Image R., 2014]
            Eigen::Matrix3d Q;
            Eigen::Vector3d e1,e2;
            if(m.norm() < L3D_EPS)
            {
                // compute nullspace of l'
                Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(l.transpose());
                Eigen::MatrixXd e = lu_decomp.kernel();

                e1 = Eigen::Vector3d(e(0,0),e(1,0),e(2,0));
                e2 = Eigen::Vector3d(e(0,1),e(1,1),e(2,1));
            }
            else
            {
                e1 = m.normalized();
                e2 = (l.cross(m)).normalized();
            }

            Q(0,0) = l(0); Q(0,1) = e1(0); Q(0,2) = e2(0);
            Q(1,0) = l(1); Q(1,1) = e1(1); Q(1,2) = e2(1);
            Q(2,0) = l(2); Q(2,1) = e1(2); Q(2,2) = e2(2);

            Eigen::Matrix3d sx = (Q-Eigen::MatrixXd::Identity(3,3))*((Q+Eigen::MatrixXd::Identity(3,3)).inverse());

            Eigen::Vector3d s(sx(2,1),sx(0,2),sx(1,0));
            double omega = m.norm();

            if(isnan(s(0)) || isnan(s(1)) || isnan(s(2)) || isnan(omega))
            {
                // symmetric line coords... do not bundle
                lines[i * LINE_SIZE + 0] = -1;
                lines[i * LINE_SIZE + 1] = 0;
                lines[i * LINE_SIZE + 2] = 0;
                lines[i * LINE_SIZE + 3] = 0;
            }
            else
            {
                lines[i * LINE_SIZE + 0] = omega;
                lines[i * LINE_SIZE + 1] = s(0);
                lines[i * LINE_SIZE + 2] = s(1);
                lines[i * LINE_SIZE + 3] = s(2);
            }

            tmp_pts[i * 6 + 0] = LC.seg3D().P1().x();
            tmp_pts[i * 6 + 1] = LC.seg3D().P1().y();
            tmp_pts[i * 6 + 2] = LC.seg3D().P1().z();
            tmp_pts[i * 6 + 3] = LC.seg3D().P2().x();
            tmp_pts[i * 6 + 4] = LC.seg3D().P2().y();
            tmp_pts[i * 6 + 5] = LC.seg3D().P2().z();
        }

        typedef g2o::BlockSolver<g2o::BlockSolverTraits<4,2>> Block;

        for(size_t i=0; i<clusters3D_->size(); ++i)
        {
            std::unique_ptr<Block::LinearSolverType> linearSolver;
            linearSolver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
                    g2o::make_unique<Block>(std::move(linearSolver)));
            g2o::SparseOptimizer optimizer;
            optimizer.setAlgorithm(solver);
            optimizer.setVerbose(true);

            fitVertex* vertex = new fitVertex();
            vertex->setEstimate(Eigen::Vector4d(lines + i*LINE_SIZE));
            vertex->setId(0);
            optimizer.addVertex(vertex);

            // iterate over 2D residuals
            std::list<L3DPP::Segment2D>::const_iterator it=clusters3D_->at(i).residuals()->begin();

            for(; it!=clusters3D_->at(i).residuals()->end(); ++it)
            {
                L3DPP::Segment2D seg2D = *it;
                size_t camera_idx = cam_global2local[seg2D.camID()];  // the given ID
                L3DPP::View* v = views_[seg2D.camID()];

                // 2D line points and direction
                Eigen::Vector4f coords = v->getLineSegment2D(seg2D.segID());
                Eigen::Vector2d p1(coords.x(),coords.y());
                Eigen::Vector2d p2(coords.z(),coords.w());
                Eigen::Vector2d dir = (p2-p1).normalized();
                fitEdge* edge = new fitEdge(p1.x(),p1.y(),p2.x(),p2.y(),-dir.y(),dir.x(),cameras +
                                            camera_idx*CAM_PARAMETERS_SIZE,intrinsics + camera_idx*INTRINSIC_SIZE);
                edge->setId(i);
                edge->setVertex(0, vertex);
                edge->setMeasurement(Eigen::Vector2d(0, 0));
                edge->setInformation(Eigen::Matrix<double,2,2>::Identity());
                optimizer.addEdge(edge);
            }

            cout << "\nstart optimization" << endl;
            optimizer.initializeOptimization();
            optimizer.optimize(100);

            Eigen::Vector4d(lines + i*LINE_SIZE) = vertex->estimate();
            Eigen::Vector4d estimate = vertex->estimate();
            *(lines + i*LINE_SIZE) = estimate[0];
            *(lines + i*LINE_SIZE + 1) = estimate[1];
            *(lines + i*LINE_SIZE + 2) = estimate[2];
            *(lines + i*LINE_SIZE + 3) = estimate[3];
            cout << "\noptimization done" << endl;
        }

        // write back
        std::vector<L3DPP::LineCluster3D> clusters_copy = *clusters3D_;
        clusters3D_->clear();
        for(size_t i=0; i<clusters_copy.size(); ++i)
        {
            L3DPP::LineCluster3D LC = clusters_copy[i];

            // get final Cayley coords
            double omega = lines[i* LINE_SIZE + 0];
            Eigen::Vector3d s(lines[i * LINE_SIZE + 1],
                              lines[i * LINE_SIZE + 2],
                              lines[i * LINE_SIZE + 3]);

            // get old coords
            Eigen::Vector3d P1_old(tmp_pts[i * 6 + 0],
                                   tmp_pts[i * 6 + 1],
                                   tmp_pts[i * 6 + 2]);
            Eigen::Vector3d P2_old(tmp_pts[i * 6 + 3],
                                   tmp_pts[i * 6 + 4],
                                   tmp_pts[i * 6 + 5]);

            Eigen::Vector3d P1,P2;
            if(omega < 0.0 || fabs(omega) < L3D_EPS)
            {
                // keep original coords
                P1 = P1_old;
                P2 = P2_old;
            }
            else
            {
                // update coords
                Eigen::Matrix3d sx = Eigen::Matrix3d::Constant(0.0);
                sx(0,1) = -s.z(); sx(0,2) = s.y();
                sx(1,0) = s.z();  sx(1,2) = -s.x();
                sx(2,0) = -s.y(); sx(2,1) = s.x();

                double nm = s.x()*s.x()+s.y()*s.y()+s.z()*s.z();
                Eigen::Matrix3d Q = 1.0/(1.0+nm) * ((1.0-nm)*Eigen::Matrix3d::Identity() + 2.0*sx + 2.0*s*s.transpose());

                Eigen::Vector3d l(Q(0,0),Q(1,0),Q(2,0));
                Eigen::Vector3d m(Q(0,1),Q(1,1),Q(2,1));
                m *= omega;

                // convert back to P1,P2
                if(fabs(l.x()) > L3D_EPS || fabs(l.y()) > L3D_EPS || fabs(l.z()) > L3D_EPS)
                {
                    Eigen::Vector3d Pm = 0.5*(P1_old+P2_old);

                    double x1,x2,x3;
                    if(fabs(l.x()) > fabs(l.y()) && fabs(l.x()) > fabs(l.z()))
                    {
                        x1 = Pm.x();
                        x3 = (-m.y()-x1*l.z())/-l.x();
                        x2 = (m.z()-x1*l.y())/-l.x();
                    }
                    else if(fabs(l.y()) > fabs(l.x()) && fabs(l.y()) > fabs(l.z()))
                    {
                        x2 = Pm.y();
                        x3 = (m.x()-x2*l.z())/-l.y();
                        x1 = (m.z()+x2*l.x())/l.y();
                    }
                    else
                    {
                        x3 = Pm.z();
                        x2 = (m.x()+x3*l.y())/l.z();
                        x1 = (-m.y()+x3*l.x())/l.z();
                    }

                    Pm = Eigen::Vector3d(x1,x2,x3);
                    P1 = Pm+l;
                    P2 = Pm-l;
                }
                else
                {
                    // numerically unstable... keep unoptimized
                    P1 = P1_old;
                    P2 = P2_old;
                }
            }

            // check length
            if((P1-P2).norm() > L3D_EPS)
            {
                // still valid
                LC.update3Dline(L3DPP::Segment3D(P1,P2));
                clusters3D_->push_back(LC);
            }
        }

    }
}

#endif //L3DPP_CERES
