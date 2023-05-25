#include <iostream>
using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

#include <math.h>
#define DEG2RAD 0.01745329252f
#define RAD2DEG 57.295779513f

const int CAMERA_POS_Y = 0;     // d (cm)
const int CAMERA_POS_X = 0;     // l (cm)
const int CAMERA_POS_Z = 60;    // h (cm)
const float FOV_H = 1.3962634f; // (degree)
const float FOV_V = 1.3962634f; // (degree)

struct CameraParameters
{
    double horizontal_fov;
    int image_width;
    int image_height;
    double near_clip;
    double far_clip;
    double noise_mean;
    double noise_std_dev;
    double hack_baseline;
    double distortion_k1;
    double distortion_k2;
    double distortion_k3;
    double distortion_t1;
    double distortion_t2;
    double camera_pos_x;
    double camera_pos_y;
    double camera_pos_z;
};

CameraParameters cam_params;

#define FILLED -1

class LaneDetect
{
public:
    Mat currFrame; // stores the upcoming frame
    Mat temp;      // stores intermediate results
    Mat temp2;     // stores the final lane segments

    int diff, diffL, diffR;
    int laneWidth;
    int diffThreshTop;
    int diffThreshLow;
    int ROIrows;
    int vertical_left;
    int vertical_right;
    int vertical_top;
    int smallLaneArea;
    int longLane;
    int vanishingPt;
    float maxLaneWidth;

    // to store various blob properties
    Mat binary_image; // used for blob removal
    int minSize;
    int ratio;
    float contour_area;
    float blob_angle_deg;
    float bounding_width;
    float bounding_length;
    Size2f sz;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    RotatedRect rotated_rect;

    LaneDetect(Mat startFrame)
    {
        currFrame = Mat(800, 800, CV_8UC1, 0.0);         // initialised the image size to 320x480
        resize(startFrame, currFrame, currFrame.size()); // resize the input to required size

        temp = Mat(currFrame.rows, currFrame.cols, CV_8UC1, 0.0);  // stores possible lane markings
        temp2 = Mat(currFrame.rows, currFrame.cols, CV_8UC1, 0.0); // stores finally selected lane marks

        vanishingPt = currFrame.rows / 2;                      // for simplicity right now
        ROIrows = currFrame.rows - vanishingPt;                // rows in region of interest
        minSize = 0.00015 * (currFrame.cols * currFrame.rows); // min size of any region to be selected as lane
        maxLaneWidth = 0.025 * currFrame.cols;                 // approximate max lane width based on image size
        smallLaneArea = 7 * minSize;
        longLane = 0.3 * currFrame.rows;
        ratio = 4;

        // these mark the possible ROI for vertical lane segments and to filter vehicle glare
        vertical_left = 2 * currFrame.cols / 5;
        vertical_right = 3 * currFrame.cols / 5;
        vertical_top = 2 * currFrame.rows / 3;

        namedWindow("lane", 2);
        namedWindow("midstep", 2);
        namedWindow("currframe", 2);
        namedWindow("laneBlobs", 2);

        getLane();
    }

    void updateSensitivity()
    {
        int total = 0, average = 0;
        for (int i = vanishingPt; i < currFrame.rows; i++)
            for (int j = 0; j < currFrame.cols; j++)
                total += currFrame.at<uchar>(i, j);
        average = total / (ROIrows * currFrame.cols);
        cout << "average : " << average << endl;
    }

    void getLane()
    {
        // medianBlur(currFrame, currFrame,5 );
        //  updateSensitivity();
        // ROI = bottom half
        for (int i = vanishingPt; i < currFrame.rows; i++)
            for (int j = 0; j < currFrame.cols; j++)
            {
                temp.at<uchar>(i, j) = 0;
                temp2.at<uchar>(i, j) = 0;
            }

        imshow("currframe", currFrame);
        blobRemoval();
    }

    void markLane()
    {
        for (int i = vanishingPt; i < currFrame.rows; i++)
        {
            // IF COLOUR IMAGE IS GIVEN then additional check can be done
            //  lane markings RGB values will be nearly same to each other(i.e without any hue)

            // min lane width is taken to be 5
            laneWidth = 5 + maxLaneWidth * (i - vanishingPt) / ROIrows;
            for (int j = laneWidth; j < currFrame.cols - laneWidth; j++)
            {

                diffL = currFrame.at<uchar>(i, j) - currFrame.at<uchar>(i, j - laneWidth);
                diffR = currFrame.at<uchar>(i, j) - currFrame.at<uchar>(i, j + laneWidth);
                diff = diffL + diffR - abs(diffL - diffR);

                // 1 right bit shifts to make it 0.5 times
                diffThreshLow = currFrame.at<uchar>(i, j) >> 1;
                // diffThreshTop = 1.2*currFrame.at<uchar>(i,j);

                // both left and right differences can be made to contribute
                // at least by certain threshold (which is >0 right now)
                // total minimum Diff should be atleast more than 5 to avoid noise
                if (diffL > 0 && diffR > 0 && diff > 5)
                    if (diff >= diffThreshLow /*&& diff<= diffThreshTop*/)
                        temp.at<uchar>(i, j) = 255;
            }
        }
    }

    void blobRemoval()
    {
        markLane();

        // find all contours in the binary image
        temp.copyTo(binary_image);
        findContours(binary_image, contours,
                     hierarchy, RETR_CCOMP,
                     CHAIN_APPROX_SIMPLE);

        // for removing invalid blobs
        if (!contours.empty())
        {
            for (size_t i = 0; i < contours.size(); ++i)
            {
                //====conditions for removing contours====//

                contour_area = contourArea(contours[i]);

                // blob size should not be less than lower threshold
                if (contour_area > minSize)
                {
                    rotated_rect = minAreaRect(contours[i]);
                    sz = rotated_rect.size;
                    bounding_width = sz.width;
                    bounding_length = sz.height;

                    // openCV selects length and width based on their orientation
                    // so angle needs to be adjusted accordingly
                    blob_angle_deg = rotated_rect.angle;
                    if (bounding_width < bounding_length)
                        blob_angle_deg = 90 + blob_angle_deg;

                    // if such big line has been detected then it has to be a (curved or a normal)lane
                    if (bounding_length > longLane || bounding_width > longLane)
                    {
                        drawContours(currFrame, contours, i, Scalar(255), FILLED, 8);
                        drawContours(temp2, contours, i, Scalar(255), FILLED, 8);
                    }

                    // angle of orientation of blob should not be near horizontal or vertical
                    // vertical blobs are allowed only near center-bottom region, where centre lane mark is present
                    // length:width >= ratio for valid line segments
                    // if area is very small then ratio limits are compensated
                    else if ((blob_angle_deg < -10 || blob_angle_deg > -10) &&
                             ((blob_angle_deg > -70 && blob_angle_deg < 70) ||
                              (rotated_rect.center.y > vertical_top &&
                               rotated_rect.center.x > vertical_left && rotated_rect.center.x < vertical_right)))
                    {

                        if ((bounding_length / bounding_width) >= ratio || (bounding_width / bounding_length) >= ratio || (contour_area < smallLaneArea && ((contour_area / (bounding_width * bounding_length)) > .75) && ((bounding_length / bounding_width) >= 2 || (bounding_width / bounding_length) >= 2)))
                        {
                            drawContours(currFrame, contours, i, Scalar(255), FILLED, 8);
                            drawContours(temp2, contours, i, Scalar(255), FILLED, 8);
                        }
                    }
                }
            }
        }
        imshow("midstep", temp);
        imshow("laneBlobs", temp2);
        imshow("lane", currFrame);
    }

    void nextFrame(Mat &nxt)
    {
        // currFrame = nxt;                        //if processing is to be done at original size

        resize(nxt, currFrame, currFrame.size()); // resizing the input image for faster processing
        getLane();
    }

    Mat getResult()
    {
        return temp2;
    }

}; // end of class LaneDetect

void build_ipm_table(
    const int srcw,
    const int srch,
    const int dstw,
    const int dsth,
    const int vptx,
    const int vpty,
    int *maptable)
{
    const float alpha_h = 0.5f * cam_params.horizontal_fov * DEG2RAD;
    const float alpha_v = 0.5f * FOV_V;
    const float gamma = -(float)(vptx - (srcw >> 1)) * alpha_h / (srcw >> 1); // camera pan angle
    const float theta = -(float)(vpty - (srch >> 1)) * alpha_v / (srch >> 1); // camera tilt angle

    const int front_map_start_position = dsth >> 1;
    const int front_map_end_position = front_map_start_position + dsth;
    const int side_map_mid_position = dstw >> 1;
    // scale to get better mapped image
    const int front_map_scale_factor = 4; // the scale 4 is for 640x480 image
    const int side_map_scale_factor = 2;

    for (int y = 0; y < dstw; ++y)
    {
        for (int x = front_map_start_position; x < front_map_end_position; ++x)
        {
            int idx = y * dsth + (x - front_map_start_position);

            int deltax = front_map_scale_factor * (front_map_end_position - x - CAMERA_POS_X);
            int deltay = side_map_scale_factor * (y - side_map_mid_position - CAMERA_POS_Y);

            if (deltay == 0)
            {
                maptable[idx] = maptable[idx - dsth];
            }
            else
            {
                int u = (int)((atan(CAMERA_POS_Z * sin(atan((float)deltay / deltax)) / deltay) - (theta - alpha_v)) / (2 * alpha_v / srch));
                int v = (int)((atan((float)deltay / deltax) - (gamma - alpha_h)) / (2 * alpha_h / srcw));
                if (u >= 0 && u < srch && v >= 0 && v < srcw)
                {
                    maptable[idx] = srcw * u + v;
                }
                else
                {
                    maptable[idx] = -1;
                }
            }
        }
    }
}

void inverse_perspective_mapping(
    const int dstw,
    const int dsth,
    const unsigned char *src,
    const int *maptable,
    unsigned char *dst)
{
    // dst image (1cm/pixel)
    int idx = 0;
    for (int j = 0; j < dsth; ++j)
    {
        for (int i = 0; i < dstw; ++i)
        {
            if (maptable[idx] != -1)
            {
                dst[i * dsth + j] = src[maptable[idx]];
            }
            else
            {
                dst[i * dsth + j] = 0;
            }
            ++idx;
        }
    }
}

int main(
    int ac,
    char **av)
{
    const int SRC_RESIZED_WIDTH = 800;
    const int SRC_RESIZED_HEIGHT = 800;
    const int DST_REMAPPED_WIDTH = 800;
    const int DST_REMAPPED_HEIGHT = 800;

    // init vanishing point at center of image
    int vanishing_point_x = SRC_RESIZED_WIDTH >> 1;
    int vanishing_point_y = SRC_RESIZED_HEIGHT >> 1;

    // build inverse perspective mapping table first
    int *ipm_table = new int[DST_REMAPPED_WIDTH * DST_REMAPPED_HEIGHT];
    build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT,
                    DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT,
                    vanishing_point_x, vanishing_point_y, ipm_table);

    // VideoCapture cap;
    // cap.open(0);
    // if (!cap.isOpened())
    // {
    //     cout << "failed to open video" << endl;
    //     return -1;
    // }

    // use image from 000024.jpg
    // Mat im = imread("../000024.jpg");
    Mat im;
    Mat imresize;
    Mat grayresize;
    Mat imremapped = Mat(DST_REMAPPED_HEIGHT, DST_REMAPPED_WIDTH, CV_8UC1);
    char key = 0;

    // video
    VideoCapture cap("../output.avi");

    while (key != 27) // press esc to stop
    {
        cap >> im;
        if (im.empty())
        {
            // End of the video reached, loop back to the beginning
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        resize(im, imresize, Size(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT));
        cvtColor(imresize, grayresize, COLOR_BGR2GRAY);

        inverse_perspective_mapping(DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT, grayresize.data, ipm_table, imremapped.data);

        line(imresize, Point(vanishing_point_x + 10, vanishing_point_y), Point(vanishing_point_x - 10, vanishing_point_y), Scalar(0, 0, 255));
        line(imresize, Point(vanishing_point_x, vanishing_point_y + 10), Point(vanishing_point_x, vanishing_point_y - 10), Scalar(0, 0, 255));

        LaneDetect detect(imremapped);

        Mat lane;

        // to gray
        detect.nextFrame(imremapped);
        imshow("resize", imresize);
        imshow("remap", imremapped);

        key = waitKey(10);
        // adjust vanishing point position
        switch (key)
        {
        case 'a':
            vanishing_point_x -= 2;
            build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT,
                            DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT,
                            vanishing_point_x, vanishing_point_y, ipm_table);
            break;
        case 'w':
            vanishing_point_y -= 2;
            build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT,
                            DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT,
                            vanishing_point_x, vanishing_point_y, ipm_table);
            break;
        case 's':
            vanishing_point_y += 2;
            build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT,
                            DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT,
                            vanishing_point_x, vanishing_point_y, ipm_table);
            break;
        case 'd':
            vanishing_point_x += 2;
            build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT,
                            DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT,
                            vanishing_point_x, vanishing_point_y, ipm_table);
            break;
        default:
            break;
        }
    }

    delete[] ipm_table;
    return 0;
}