#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

#include <math.h>
#define DEG2RAD 0.01745329252f
#define RAD2DEG 57.295779513f

struct CameraParameters
{
    double horizontal_fov; // in radian
    double vertical_fov;   // in radian
    int image_width;       // in pixels
    int image_height;      // in pixels
    double near_clip;      // in meters
    double far_clip;       // in meters
    double noise_mean;     // in meters
    double noise_std_dev;  // in meters
    double hack_baseline;  // in meters
    double distortion_k1;  // radial distortion coefficient
    double distortion_k2;  // radial distortion coefficient
    double distortion_k3;  // radial distortion coefficient
    double distortion_t1;  // tangential distortion coefficient
    double distortion_t2;  // tangential distortion coefficient
    double camera_pos_x;   // in cm
    double camera_pos_y;   // in cm
    double camera_pos_z;   // in cm
    double cam_scale_x;    // in cm
    double cam_scale_y;    // in cm
};

CameraParameters cam_params;
int *maptable = new int[800 * 800];
int *maptable_m = new int[800 * 800];

void Init();
void LogParams();
void LogMaptable();
int PxToCm(int p);
int CmToPx(int p);
void BuildIPMTable(const int src_w, const int src_h, const int dst_w, const int dst_h, const int vanishing_pt_x, const int vanishing_pt_y, int *maptable);
void MaptablePxToM(int *maptable, int maptable_size, int *maptable_m);
void InversePerspective(const int dst_w, const int dst_h, const unsigned char *src, const int *maptable, unsigned char *dst);

int main()
{
    const int DST_REMAPPED_WIDTH = 800;
    const int DST_REMAPPED_HEIGHT = 800;
    Init();
    LogParams();
    BuildIPMTable(cam_params.image_width, cam_params.image_height, cam_params.image_width, cam_params.image_height, cam_params.image_width >> 1, cam_params.image_height >> 1, maptable);
    // LogMaptable();
    MaptablePxToM(maptable, cam_params.image_width * cam_params.image_height, maptable_m);

    printf("Passed test 1\n");

    Mat frame;
    Mat frame_resize;
    Mat frame_gray_resize;
    Mat frame_remapped = Mat(DST_REMAPPED_HEIGHT, DST_REMAPPED_WIDTH, CV_8UC1);

    printf("Passed test 2\n");

    char key = 0;

    VideoCapture cap("../output.avi");

    while (key != 27)
    {
        cap >> frame;
        if (frame.empty())
        {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        resize(frame, frame_resize, Size(cam_params.image_width, cam_params.image_height));
        cvtColor(frame_resize, frame_gray_resize, COLOR_BGR2GRAY);

        InversePerspective(DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT, frame_gray_resize.data, maptable, frame_remapped.data);

        line(frame, Point(cam_params.image_width >> 1, 0), Point(cam_params.image_width >> 1, cam_params.image_height), Scalar(0, 0, 255), 1);
        line(frame, Point(0, cam_params.image_height >> 1), Point(cam_params.image_width, cam_params.image_height >> 1), Scalar(0, 0, 255), 1);

        imshow("frame", frame);
        imshow("frame_remapped", frame_remapped);
        waitKey(1);
    }
}

void Init()
{
    cam_params.horizontal_fov = 1.3962634; // rad
    cam_params.vertical_fov = 2 * atan(tan(cam_params.horizontal_fov / 2) * 1);
    cam_params.image_width = 800;
    cam_params.image_height = 800;
    cam_params.near_clip = 0.1;
    cam_params.near_clip = 0.02;
    cam_params.far_clip = 300;
    cam_params.noise_mean = 0.0;
    cam_params.noise_std_dev = 0.007;
    cam_params.hack_baseline = 0.07;
    cam_params.distortion_k1 = 0.0;
    cam_params.distortion_k2 = 0.0;
    cam_params.distortion_k3 = 0.0;
    cam_params.distortion_t1 = 0.0;
    cam_params.distortion_t2 = 0.0;
    cam_params.camera_pos_x = 75; // cm
    cam_params.camera_pos_y = 1;
    cam_params.camera_pos_z = 202.5;
    cam_params.cam_scale_x = (2 * cam_params.camera_pos_x * tan(cam_params.horizontal_fov / 2)) / cam_params.image_width;
    cam_params.cam_scale_y = (2 * cam_params.camera_pos_y * tan(cam_params.vertical_fov / 2)) / cam_params.image_height;
}

void LogParams()
{
    printf("\n                       Camera parameters                      \n");
    printf("==============================================================\n");
    printf("Horizontal FOV      : %f\n", cam_params.horizontal_fov);
    printf("Vertical FOV        : %f\n", cam_params.vertical_fov);
    printf("Image width         : %d\n", cam_params.image_width);
    printf("Image height        : %d\n", cam_params.image_height);
    printf("Near clip           : %f\n", cam_params.near_clip);
    printf("Far clip            : %f\n", cam_params.far_clip);
    printf("Noise mean          : %f\n", cam_params.noise_mean);
    printf("Noise std dev       : %f\n", cam_params.noise_std_dev);
    printf("Hack baseline       : %f\n", cam_params.hack_baseline);
    printf("Distortion k1       : %f\n", cam_params.distortion_k1);
    printf("Distortion k2       : %f\n", cam_params.distortion_k2);
    printf("Distortion k3       : %f\n", cam_params.distortion_k3);
    printf("Distortion t1       : %f\n", cam_params.distortion_t1);
    printf("Distortion t2       : %f\n", cam_params.distortion_t2);
    printf("Camera position x   : %f\n", cam_params.camera_pos_x);
    printf("Camera position y   : %f\n", cam_params.camera_pos_y);
    printf("Camera position z   : %f\n", cam_params.camera_pos_z);
    printf("Camera scale x      : %f\n", cam_params.cam_scale_x);
    printf("Camera scale y      : %f\n", cam_params.cam_scale_y);
    printf("==============================================================\n");
    printf("\n");
}

//============================================================================================================================

void BuildIPMTable(const int src_w, const int src_h, const int dst_w, const int dst_h, const int vanishing_pt_x, const int vanishing_pt_y, int *maptable)
{
    float alpha = cam_params.horizontal_fov / 2;
    float gamma = -(float)(vanishing_pt_x - (src_w >> 1)) * alpha / (src_w >> 1);
    float theta = -(float)(vanishing_pt_y - (src_h >> 1)) * alpha / (src_h >> 1);

    int front_map_pose_start = (dst_h >> 1) - 100;
    int front_map_pose_end = front_map_pose_start + dst_h;

    int side_map_mid_pose = dst_w >> 1;

    // int front_map_scale = cam_params.cam_scale_y;
    // int side_map_scale = cam_params.cam_scale_x;

    int front_map_scale = 2;
    int side_map_scale = 2;

    for (int y = 0; y < dst_w; ++y)
    {
        for (int x = front_map_pose_start; x < front_map_pose_end; ++x)
        {
            int index = y * dst_h + (x - front_map_pose_start);

            int delta_x = front_map_scale * (front_map_pose_end - x - cam_params.camera_pos_x);
            int delta_y = front_map_scale * (y - side_map_mid_pose - cam_params.camera_pos_y);

            if (!delta_y)
                maptable[index] = maptable[index - dst_h];
            else
            {
                int u = (int)((atan(cam_params.camera_pos_z * sin(atan((float)delta_y / delta_x)) / delta_y) - (theta - alpha)) / (2 * alpha / src_h));
                int v = (int)((atan((float)delta_y / delta_x) - (gamma - alpha)) / (2 * alpha / src_w));

                if (u >= 0 && u < src_h && v >= 0 && v < src_w)
                    maptable[index] = src_w * u + v;
                else
                    maptable[index] = -1;
            }
        }
    }
}

void LogMaptable()
{
    for (int i = 0; i < 800 * 800; i++)
    {
        printf("Maptable[%d] => %d\n", i, maptable[i]);
    }
}

void InversePerspective(const int dst_w, const int dst_h, const unsigned char *src, const int *maptable, unsigned char *dst)
{
    int index = 0;
    for (int j = 0; j < dst_h; ++j)
    {
        for (int i = 0; i < dst_w; ++i)
        {
            if (maptable[index] == -1)
            {
                dst[i * dst_h + j] = 0;
            }
            else
            {
                dst[i * dst_h + j] = src[maptable[index]];
            }
            ++index;
        }
    }
}

void MaptablePxToM(int *maptable, int maptable_size, int *maptable_m)
{
    for (int i = 0; i < maptable_size; ++i)
    {
        maptable_m[i] = (int)(maptable[i] * cam_params.cam_scale_x / 100);
    }
}

int PxToCm(int p)
{
    return (int)(p * cam_params.cam_scale_x);
}

int CmToPx(int p)
{
    return (int)(p / cam_params.cam_scale_x);
}