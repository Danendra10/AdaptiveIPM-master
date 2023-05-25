#include "opencv2/opencv.hpp"

struct InversePerspectiveMapping
{
    cv::Mat K; // Intrinsic matrix
    cv::Mat R; // Rotation matrix
    cv::Mat T; // Translation matrix

    int frameWidth;  // Width of the frame
    int frameHeight; // Height of the frame

    // ROI parameters
    int roiStartX; // Start x-coordinate of the ROI
    int roiStartY; // Start y-coordinate of the ROI
    int roiWidth;  // Width of the ROI
    int roiHeight; // Height of the ROI

    // Constructor
    InversePerspectiveMapping(double fx, double fy, double cx, double cy, double Tx, double Ty,
                              int frameWidth, int frameHeight,
                              int roiStartX, int roiStartY, int roiWidth, int roiHeight)
    {
        K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        R = cv::Mat::eye(3, 3, CV_64F);
        T = (cv::Mat_<double>(3, 1) << Tx, Ty, 0);

        this->frameWidth = frameWidth;
        this->frameHeight = frameHeight;

        this->roiStartX = roiStartX;
        this->roiStartY = roiStartY;
        this->roiWidth = roiWidth;
        this->roiHeight = roiHeight;
    }

    cv::Point3d convertToBirdsEyeView(int x, int y, double cameraX, double cameraY, double cameraZ)
    {
        // Convert pixel coordinates to ROI coordinates
        double roiX = static_cast<double>(x - roiStartX);
        double roiY = static_cast<double>(y - roiStartY);

        // Convert ROI coordinates to normalized coordinates in the range [0, 1]
        double normalizedX = roiX / roiWidth;
        double normalizedY = roiY / roiHeight;

        // Convert normalized coordinates to pixel coordinates in the frame
        double frameX = normalizedX * frameWidth;
        double frameY = normalizedY * frameHeight;

        // Perform inverse perspective mapping
        cv::Mat P = K * cv::Mat::eye(3, 4, CV_64F);
        cv::Mat inverseP;
        cv::invert(P, inverseP, cv::DECOMP_SVD);

        cv::Mat point2D = (cv::Mat_<double>(3, 1) << frameX, frameY, 1);
        cv::Mat point3D = inverseP * point2D;

        // Normalize the 3D point
        double X = point3D.at<double>(0) / point3D.at<double>(2);
        double Y = point3D.at<double>(1) / point3D.at<double>(2);
        double Z = point3D.at<double>(2) / point3D.at<double>(2);

        // for (int i = 0; i < point3D.rows; i++)
        // {
        //     for (int j = 0; j < point3D.cols; j++)
        //     {
        //         std::cout << i << " " << j << " " << point3D.at<double>(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // Convert to bird's eye view
        double birdX = X + cameraX;
        double birdY = Y + cameraY;
        double birdZ = Z + cameraZ;

        return cv::Point3d(birdX, birdY, birdZ);
    }

    void displayBirdsEyeView(const cv::Mat &frame, double cameraX, double cameraY, double cameraZ)
    {
        cv::Mat birdsEyeView(frame.size(), frame.type());

        // Iterate over all pixels in the frame
        for (int y = 0; y < frame.rows; ++y)
        {
            for (int x = 0; x < frame.cols; ++x)
            {
                // Convert pixel coordinates to bird's eye view
                cv::Point3d result = convertToBirdsEyeView(x, y, cameraX, cameraY, cameraZ);

                // Map the bird's eye view coordinates back to the frame
                int birdX = static_cast<int>(result.x);
                int birdY = static_cast<int>(result.y);

                printf("%d %d\n", birdX, birdY);

                // Ensure the mapped coordinates are within the frame boundaries
                if (birdX >= 0 && birdX < frame.cols && birdY >= 0 && birdY < frame.rows)
                {
                    // Set the pixel value in the bird's eye view frame
                    birdsEyeView.at<cv::Vec3b>(y, x) = frame.at<cv::Vec3b>(birdY, birdX);
                }
            }
        }

        // Display the bird's eye view frame
        cv::imshow("Bird's Eye View", birdsEyeView);
        cv::imshow("Original", frame);
        cv::waitKey(0);
    }

    void getROIFrame(cv::Mat input_frame, cv::Mat &output_frame)
    {
        output_frame = input_frame(cv::Rect(roiStartX, roiStartY, roiWidth, roiHeight));
    }
};

int main()
{
    // Define the camera parameters
    double fx = 476.7030836014194;
    double fy = 476.7030836014194;
    double cx = 400.5;
    double cy = 400.5;
    double Tx = -33.36921585209936;
    double Ty = 0;

    // Define the frame size
    int frameWidth = 800;
    int frameHeight = 800;

    // Define the ROI parameters
    int roiStartX = 0;
    int roiStartY = 400;
    int roiWidth = 800;
    int roiHeight = 400;

    // Create the inverse perspective mapping object
    InversePerspectiveMapping ipm(fx, fy, cx, cy, Tx, Ty,
                                  frameWidth, frameHeight,
                                  roiStartX, roiStartY, roiWidth, roiHeight);

    // Define the camera position
    double cameraX = 0.75;
    double cameraY = 0;
    double cameraZ = 2.025;

    cv::VideoCapture cap("../output.avi");

    // Iterate over all x and y values in the ROI
    // for (int y = roiStartY; y < roiStartY + roiHeight; ++y)
    // {
    //     for (int x = roiStartX; x < roiStartX + roiWidth; ++x)
    //     {
    //         // Convert pixel coordinates to bird's eye view
    //         cv::Point3d result = ipm.convertToBirdsEyeView(x, y, cameraX, cameraY, cameraZ);

    //         // Print the result
    //         std::cout << "Pixel Coordinates: (" << x << ", " << y << ") => Bird's Eye View Coordinates: (" << result.x << ", " << result.y << ", " << result.z << ")" << std::endl;
    //     }
    // }

    while (1)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
            break;
        ipm.getROIFrame(frame, frame);
        ipm.displayBirdsEyeView(frame, cameraX, cameraY, cameraZ);
    }

    return 0;
}
