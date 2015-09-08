#ifndef OPENCV_UTILS
#define OPENCV_UTILS

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <math.h>
#include <functional>

class OpencvUtils {

private:
    struct CallbackData {
        cv::Mat image;
        cv::Mat result;
        int slider1 = 0;
        int slider2 = 0;
        int type = 0;
        int type2 = 0;
        std::vector<std::vector<cv::Point>> contours;
    };

    static void trackbar_threshold(CallbackData& data);
    static void trackbar_canny(CallbackData& data);
    static void trackbar_adaptative_threshold(CallbackData& data);
    static void trackbar_find_contours(CallbackData& data);


    static void callback_threshold(int value, void *ptr);
    static void callback_canny(int value, void *ptr);
    static void callback_adaptative_threshold(int value, void *ptr);
    static void callback_find_contours(int value, void *ptr);

public:
    OpencvUtils();

    void create_trackbar(cv::Mat& src, cv::Mat& dst, std::string method,
    		int type = -1);

    static void show_image(std::string name, const cv::Mat& image);

    template<typename Type>
    static void swap(Type& var1, Type& var2);

    void adaptative_threshold_on_sliced_image(cv::Mat& image,
    		int number_of_slices,  int max_threshold = 255);
    void cut_image(cv::Mat& image, int number_of_slices,
    		std::vector<std::vector<cv::Mat>>& slices);
    void join_image(std::vector<std::vector<cv::Mat>>& slices, cv::Mat& image);
    void get_rois(cv::Mat& image, int number_of_slices,
    		std::vector<std::vector<cv::Rect>>& rois);

    template<typename Type>
    static Type point_between(Type p0, Type p1, float percentage);
    static cv::Point2f calculate_coordinate_distance(cv::Point2f p0,
    		cv::Point2f p1, bool to_zero = false);
    static float calculate_distance(cv::Point2f p0, cv::Point2f p1,
    		bool to_zero = false);
    static void calculate_rotated_rectangle_size(cv::Point2f points[4],
    		float& width, float& height);
    static void calculate_rotated_rectangle_size_2(cv::Point2f points[4],
    		float& width, float& height);

    static int get_index_large_contour(std::vector<std::vector<cv::Point>> contours);
    static void delete_large_contour(std::vector<std::vector<cv::Point>>& contours);
    static void save_only_large_contour(std::vector<std::vector<cv::Point>>& contours);
    static void delete_points_outside_figure(std::vector<std::vector<cv::Point>>& contour,
    		std::vector<cv::Point>& figure);

    static void get_points_on_edges(std::vector<cv::Point>& figure,
    		std::vector<cv::Point>& contour,
			std::vector<cv::Point>& points_on_edges);
    static void get_perpendicular_line(cv::Mat& image, cv::Point2f& src0,
    		cv::Point2f& src1, cv::Point2f point,
			cv::Point2f& dst0, cv::Point2f& dst1);
    static void get_paralell_line(cv::Mat& image, cv::Point2f& src0,
    		cv::Point2f& src1, cv::Point2f& point, cv::Point2f& dst0,
			cv::Point2f& dst1);
    static void correct_perspective(const cv::RotatedRect& rect,
    		const cv::Mat& src, cv::Mat& dst);

    static bool is_grey(cv::Vec3b color, int threshold = 30);
    static void black_or_whites(cv::Mat& image);
    static void save_only_whites(cv::Mat& image, cv::Mat& result,
    		int threshold = 30);
    static void fixed_image_whith_marker(cv::Mat& original, cv::Mat& marker);
    static float count_color_pixels(const cv::Mat& image, cv::Point p0,
    		cv::Point p1, cv::Vec3b color);
    static float count_no_color_pixels(const cv::Mat& image, cv::Point p0,
    		cv::Point p1, cv::Vec3b color);
    static bool found_color(cv::Mat& image, cv::Vec3b color, cv::Point2f& p0,
    		cv::Point2f& p1);

    template<typename Type>
    static void draw_lines(cv::Mat& drawing, std::vector<Type>& lines,
    		const cv::Scalar& color);
    static void draw_fill_polygon(cv::Mat& image, std::vector<cv::Point>& polygon,
    		const cv::Scalar& color);
    static int draw_large_contour(cv::Mat& drawing,
    		std::vector<std::vector<cv::Point>>& contours,
			const cv::Scalar& color = cv::Scalar(255, 255, 255));
    static void draw_all_contours(cv::Mat& drawing,
    		std::vector<std::vector<cv::Point>>& contours,
			const cv::Scalar& color);
    static void draw_color_outside_contour(cv::Mat& image,
    		std::vector<cv::Point> contour, cv::Vec3b color);
    static void draw_color_outside_polygon(cv::Mat& image,
    		std::vector<cv::Point> polygon, cv::Vec3b color);
    static void draw_all_contours(cv::Mat& drawing,
    		std::vector<std::vector<cv::Point>>& contours);
    static void draw_borders(cv::Mat& image, cv::Scalar color = CV_RGB(255, 255, 255));

    static void min_rotated_rect(const std::vector<std::vector<cv::Point>>& contours,
    		cv::RotatedRect& minRect);
    static void min_rotated_rect_in_large_contour(
    		const std::vector<std::vector<cv::Point>>& contours,
			cv::RotatedRect& minRect);

    static void cut_rectangle2(cv::Mat& dst, cv::RotatedRect& rect, float percentage);
    static void cut_rectangle(const cv::RotatedRect& rect, const cv::Mat& image,
    		cv::Mat& cropped);
    static void rotate(cv::Mat& src, double angle, cv::Mat& dst);

    static float calculate_height_width_ratio(const cv::Mat& image);
    static float calculate_height_width_ratio(const cv::Rect& rectangle);
    static float calculate_height_width_ratio(const cv::RotatedRect& rectangle);

    static bool closer_to_border(cv::Mat& image, cv::Point2f& p0, cv::Point2f& p1);

    template<typename Type>
    static bool point_inside(cv::Mat& image, Type& p);
    static void cut_line_to_border(cv::Mat& image, cv::Point2f& p1, cv::Point2f& p2);
    static void expand_to_border(cv::Mat& image, cv::Point2f& p0, cv::Point2f& p1);
    static void expand_line(cv::Mat& image, cv::Point2f& p1, cv::Point2f& p2,
    		char type);

    static void equation_of_line(cv::Point2f& p0, cv::Point2f& p1,
    		cv::Vec3f& equation);
    static void bisector_two_lines(cv::Point2f& p0, cv::Point2f& p1,
    		cv::Point2f& p2, cv::Point2f& p3, cv::Vec3f& equation_result,
			bool type = true);
    static void bisector_two_lines(cv::Vec3f& equation0, cv::Vec3f& equation1,
    		cv::Vec3f& equation_result, bool type = true);
    static void intersection_of_lines(cv::Vec3f& equation0,
    		cv::Vec3f& equation1, cv::Point2f& intersection_point);
    static void intersection_of_bisectors_inside_image(cv::Mat& image,
    		cv::Point2f& p0, cv::Point2f& p1,
			cv::Point2f& p2, cv::Point2f& p3,
			cv::Point2f& intersection, float& radius);
    static void intersection_of_bisectors_inside_image(cv::Mat&,
    		cv::Vec3f& equation0, cv::Vec3f& equation1,
			cv::Vec3f& equation2, cv::Point2f& intersection,
			float& radius);

    static float distance_from_point_to_rect(cv::Point2f& point, cv::Vec3f& rect);
    static void point_minimal_distance_from_point_to_rect(cv::Point2f& point,
    		cv::Vec3f& rect, cv::Point2f& solution);
    static float get_variance(cv::Mat& image);
};

#endif
