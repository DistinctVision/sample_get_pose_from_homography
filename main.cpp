#include <iostream>
#include <vector>
#include <utility>
#include <cassert>

#include <Eigen/Eigen>
#include <Eigen/SVD>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>

using namespace std;
using namespace Eigen;

struct Pose
{
    Matrix3d R;
    Vector3d t;
};

Matrix3d computeHomography(const vector<cv::Point2f> & imagePoints,
                           const vector<Vector2d, Eigen::aligned_allocator<Vector2d>> & planePoints)
{
    assert(planePoints.size() >= 4);
    assert(imagePoints.size() == planePoints.size());

    Eigen::MatrixXd M(std::max(static_cast<int>(planePoints.size() * 2), 9), 9);
    for (std::size_t i = 0; i < planePoints.size(); ++i)
    {
        int offset = static_cast<int>(i * 2);

        const Vector2d & planePoint = planePoints[i];
        const cv::Point2f & imagePoint = imagePoints[i];

        M(offset, 0) = planePoint.x();
        M(offset, 1) = planePoint.y();
        M(offset, 2) = 1;
        M(offset, 3) = 0;
        M(offset, 4) = 0;
        M(offset, 5) = 0;
        M(offset, 6) = - planePoint.x() * static_cast<double>(imagePoint.x);
        M(offset, 7) = - planePoint.y() * static_cast<double>(imagePoint.x);
        M(offset, 8) = - static_cast<double>(imagePoint.x);

        ++offset;

        M(offset, 0) = 0;
        M(offset, 1) = 0;
        M(offset, 2) = 0;
        M(offset, 3) = planePoint.x();
        M(offset, 4) = planePoint.y();
        M(offset, 5) = 1;
        M(offset, 6) = - planePoint.x() * static_cast<double>(imagePoint.y);
        M(offset, 7) = - planePoint.y() * static_cast<double>(imagePoint.y);
        M(offset, 8) = - static_cast<double>(imagePoint.y);

        ++offset;
    }

    if (planePoints.size() == 4)
        M.row(8).setZero();

    JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinV);
    Matrix<double, 9, 1> h = svd.matrixV().col(8);

    Matrix3d H;
    H << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
    return H;
}

Pose getMarkerPose(const Matrix3d & K,
                   const vector<cv::Point2f> & imagePoints,
                   const vector<Vector2d, Eigen::aligned_allocator<Vector2d>> & planePoints)
{
    Matrix3d H = computeHomography(imagePoints, planePoints);
    Matrix3d Rt = K.inverse() * H;

    Vector3d r_x = Rt.col(0);
    Vector3d r_y = Rt.col(1);
    Vector3d t = Rt.col(2);
    double scale = (r_x.norm() + r_y.norm()) * 0.5;

    r_x /= scale;
    r_y /= scale;
    t /= scale;

    Matrix3d R;
    R.col(0) = r_x;
    R.col(1) = r_y;
    R.col(2) = r_x.cross(r_y);

    Vector3d p = - R.inverse() * t;
    if (p.z() < 0.0)
    {
        t = - t;
        R.col(0) = - R.col(0);
        R.col(1) = - R.col(1);
    }

    return Pose { R, t };
}

void drawCube(cv::Mat frame, const Matrix3d & K, const Pose & pose, const cv::Scalar & color)
{
    static const Vector3d vertices[8] = {
        Vector3d(0.0, 0.0, 0.0),
        Vector3d(1.0, 0.0, 0.0),
        Vector3d(1.0, 1.0, 0.0),
        Vector3d(0.0, 1.0, 0.0),
        Vector3d(0.0, 0.0, 1.0),
        Vector3d(1.0, 0.0, 1.0),
        Vector3d(1.0, 1.0, 1.0),
        Vector3d(0.0, 1.0, 1.0)
    };

    static const pair<int, int> edges[12] = {
        make_pair(0, 1),
        make_pair(1, 2),
        make_pair(2, 3),
        make_pair(3, 0),
        make_pair(4, 5),
        make_pair(5, 6),
        make_pair(6, 7),
        make_pair(7, 4),
        make_pair(0, 4),
        make_pair(1, 5),
        make_pair(2, 6),
        make_pair(3, 7),
    };

    for (const pair<int, int> & edge : edges)
    {
        Vector3d hp1 = K * (pose.R * vertices[edge.first] + pose.t);
        Vector3d hp2 = K * (pose.R * vertices[edge.second] + pose.t);
        cv::Point2f p1(static_cast<float>(hp1.x() / hp1.z()), static_cast<float>(hp1.y() / hp1.z()));
        cv::Point2f p2(static_cast<float>(hp2.x() / hp2.z()), static_cast<float>(hp2.y() / hp2.z()));
        cv::line(frame, p1, p2, color, 2);
    }
}

int main()
{

    cv::VideoCapture capture;

    if (!capture.open(1))
    {
        cerr << "camera not found" << endl;
        return -1;
    }

    if (!capture.isOpened())
    {
        cerr << "camera not open" << endl;
        return -2;
    }

    cv::Mat cvFrameImage;
    if (!capture.read(cvFrameImage))
    {
        cerr << "frame not readed" << endl;
        return -3;
    }

    double pixelFocalLengthX = cvFrameImage.size().width * 1.0;
    double pixelFocalLengthY = - pixelFocalLengthX;
    double opticalCenterX = cvFrameImage.size().width * 0.5;
    double opticalCenterY = cvFrameImage.size().height * 0.5;
    Matrix3d K;
    K << pixelFocalLengthX, 0.0, opticalCenterX,
         0.0, pixelFocalLengthY, opticalCenterY,
         0.0, 0.0, 1.0;

    vector<Vector2d, Eigen::aligned_allocator<Vector2d>> markerPoints(4);
    markerPoints[0] << 0.0, 0.0;
    markerPoints[1] << 0.0, 1.0;
    markerPoints[2] << 1.0, 1.0;
    markerPoints[3] << 1.0, 0.0;

    cv::aruco::Dictionary arucoDict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);

    const vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),
        cv::Scalar(255, 155, 0),
        cv::Scalar(155, 255, 0),
        cv::Scalar(0, 255, 155),
        cv::Scalar(0, 155, 255),
        cv::Scalar(155, 0, 255),
        cv::Scalar(255, 0, 155)
    };

    while (capture.isOpened())
    {
        if (!capture.read(cvFrameImage))
        {
            break;
        }

        std::vector<std::vector<cv::Point2f>> markerCorners;
        std::vector<int> markerIDs;
        cv::aruco::detectMarkers(cvFrameImage, arucoDict, markerCorners, markerIDs);

        for (size_t i = 0; i < markerCorners.size(); ++i)
        {
            Pose pose = getMarkerPose(K, markerCorners[i], markerPoints);
            drawCube(cvFrameImage, K, pose,
                     colors[static_cast<size_t>(markerIDs[i]) % colors.size()]);
        }

        cv::imshow("frame", cvFrameImage);

        int key = cv::waitKey(33);
        if (key == 27)
        {
            break;
        }
    }
    return 0;
}
