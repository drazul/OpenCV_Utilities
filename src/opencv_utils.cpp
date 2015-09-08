#include "opencv_utils.h"

OpencvUtils::OpencvUtils() {
}

void
OpencvUtils::show_image(std::string name, const cv::Mat& image) {
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, image);
}

cv::Point2f
OpencvUtils::calculate_coordinate_distance(cv::Point2f p0, cv::Point2f p1, bool to_zero) {
	if (to_zero){
		if(p0.x < 0) p0.x = 0;
		if(p0.y < 0) p0.y = 0;
		if(p1.x < 0) p1.x = 0;
		if(p1.y < 0) p1.y = 0;
	}
	return cv::Point2f(std::abs(p0.x - p1.x), std::abs(p0.y - p1.y));
}

float
OpencvUtils::calculate_distance(cv::Point2f p0, cv::Point2f p1, bool to_zero) {
    cv::Point2f distance = calculate_coordinate_distance(p0, p1, to_zero);
    return sqrt(pow(distance.x, 2) + pow(distance.y, 2));
}

void
OpencvUtils::calculate_rotated_rectangle_size(cv::Point2f points[4],
        float& width, float& height) {

    float widthA, widthB, heightA, heightB;

    heightA = std::abs(points[1].x - points[2].x);
    heightB = std::abs(points[0].x - points[3].x);

    widthA = std::abs(points[2].y - points[3].y);
    widthB = std::abs(points[1].y - points[0].y);

    width = std::max(widthA, widthB);
    height = std::max(heightA, heightB);
}

void
OpencvUtils::calculate_rotated_rectangle_size_2(cv::Point2f points[4],
        float& width, float& height) {

    float widthA, widthB, heightA, heightB;
    heightA = calculate_distance(points[1], points[2]);
    heightB = calculate_distance(points[0], points[3]);

    widthA = calculate_distance(points[2], points[3]);
    widthB = calculate_distance(points[1], points[0]);

    width = std::max(widthA, widthB);
    height = std::max(heightA, heightB);
}

void
OpencvUtils::correct_perspective(const cv::RotatedRect& rect,
		const cv::Mat& src, cv::Mat& dst) {

    cv::Point2f points[4];
    rect.points(points);

    float width, height;
    calculate_rotated_rectangle_size(points, width, height);

    cv::Point2f final_points[4] = { cv::Point2f(0, 0), cv::Point2f(width - 1, 0),
            cv::Point2f(width - 1, height - 1), cv::Point2f(0, height - 1) };

    cv::Mat persective_transform = cv::getPerspectiveTransform(points,
            final_points);

    cv::warpPerspective(src, dst, persective_transform,
            cv::Size(width, height));
}

bool
OpencvUtils::is_grey(cv::Vec3b color, int threshold) {

    if (color(0) == color(1) && color(0) == color(2) && color(0) == 0)
        return false;

    int min_value, max_value, value;
    min_value = (int) color(0), max_value = (int) color(0);

    for (int z = 1; z < 3; z++) {
        value = (int) color(z);
        max_value = std::max(value, max_value);
        min_value = std::min(value, min_value);
    }

    return (max_value - min_value) < threshold;
}

void
OpencvUtils::black_or_whites(cv::Mat& image) {
    cv::Vec3b black = cv::Vec3b(0, 0, 0);
    cv::Vec3b white = cv::Vec3b(255, 255, 255);

    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b& color = image.at<cv::Vec3b>(i, j);

            if (color != black)
                color = white;
        }
}

void
OpencvUtils::draw_color_outside_contour(cv::Mat& image,
		std::vector<cv::Point> contour, cv::Vec3b color) {
	std::vector<cv::Point> figure;
	for (auto& p: contour)
		figure.push_back(cv::Point(p.y, p.x));

	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
			if(cv::pointPolygonTest(figure, cv::Point(i, j), false) <= 0)
				image.at<cv::Vec3b>(i, j) = color;
}

void
OpencvUtils::draw_color_outside_polygon(cv::Mat& image,
		std::vector<cv::Point> polygon, cv::Vec3b color) {

	//I don't know why but we need swap x and y in contour points
	std::vector<cv::Point> figure;
	figure.push_back(cv::Point(polygon[0].y, polygon[0].x));
	figure.push_back(cv::Point(polygon[1].y, polygon[1].x));
	figure.push_back(cv::Point(polygon[2].y, polygon[2].x));
	figure.push_back(cv::Point(polygon[3].y, polygon[3].x));

    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            if(cv::pointPolygonTest(figure, cv::Point(i, j), false) <= 0)
                image.at<cv::Vec3b>(i, j) = color;
}

void
OpencvUtils::save_only_whites(cv::Mat& image, cv::Mat& result,
		int threshold) {

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b& color = image.at<cv::Vec3b>(i, j);
            cv::Vec3b& color_res = result.at<cv::Vec3b>(i, j);
            if (!is_grey(color, threshold))
            	color_res = cv::Vec3b(0, 0, 0);
        }
    }
}

void
OpencvUtils::fixed_image_whith_marker(cv::Mat& original, cv::Mat& marker) {
    for (int i = 0; i < marker.rows; i++) {
        for (int j = 0; j < marker.cols; j++) {
            cv::Vec3b& color = marker.at<cv::Vec3b>(i, j);
            if (color == cv::Vec3b(0, 0, 0))
            	original.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
        }
    }
}

void
OpencvUtils::draw_fill_polygon(cv::Mat& image, std::vector<cv::Point>& polygon,
		const cv::Scalar& color) {

	// This part not compile on windows OS, but is the same functionality
  // that can see below, but cleaner and more memory safe

	int vertex[] = { (int) polygon.size()};
	cv::Point poly[1][vertex[0]];

	std::copy(polygon.begin(), polygon.end(), poly[0]);
	const cv::Point* ppt[1] = { poly[0] };

  cv::fillPoly(image, ppt, vertex, 1, color);

/*
    int vertex[] = { (int) polygon.size()};
    cv::Point *p = new cv::Point[vertex[0]];

    std::copy(polygon.begin(), polygon.end(), p);
    const cv::Point* poly[1] = { p };
    const cv::Point* ppt[1] = { poly[0] };

    cv::fillPoly(image, ppt, vertex, 1, color);
    delete [] p;
*/
}

template<typename Type> void
OpencvUtils::draw_lines(cv::Mat& drawing,
        std::vector<Type>& lines, const cv::Scalar& color) {

    for (auto& line : lines) {
        cv::line(drawing, cv::Point(line[0], line[1]),
                cv::Point(line[2], line[3]), color, 5);
        cv::circle(drawing, cv::Point(line[0], line[1]), 2, CV_RGB(255, 0, 0));
        cv::circle(drawing, cv::Point(line[2], line[3]), 2, CV_RGB(255, 0, 0));
    }
}

int
OpencvUtils::get_index_large_contour(
        std::vector<std::vector<cv::Point>> contours) {

    int large_contour = 0;
    float area, area_max = 0;

    for (unsigned int i = 0; i < contours.size(); i++) {
        area = cv::contourArea(contours[i]);
        if (area > area_max) {
            large_contour = i;
            area_max = area;
        }
    }

    return large_contour;
}

void
OpencvUtils::save_only_large_contour(std::vector<std::vector<cv::Point>>& contours) {
	std::vector<std::vector<cv::Point>> new_contours;
	new_contours.push_back(contours[get_index_large_contour(contours)]);
	contours = new_contours;
}

void
OpencvUtils::delete_points_outside_figure(std::vector<std::vector<cv::Point>>& contour,
		std::vector<cv::Point>& figure) {

	for(unsigned int i = 0; i < contour.size(); i++)
		for(int j = contour[i].size() - 1; j >= 0; j--)
			if(pointPolygonTest(figure, contour[i][j], false) < 0)
				contour[i].erase(contour[i].begin() + j);
}

void
OpencvUtils::get_points_on_edges(std::vector<cv::Point>& figure,
		std::vector<cv::Point>& contour,
		std::vector<cv::Point>& points_on_edges) {

	for(unsigned int i = 0; i < contour.size(); i++)
		if(pointPolygonTest(figure, contour[i], false) == 0)
			points_on_edges.push_back(contour[i]);
}

void
OpencvUtils::get_paralell_line(cv::Mat& image, cv::Point2f& src0,
		cv::Point2f& src1, cv::Point2f& point, cv::Point2f& dst0,
		cv::Point2f& dst1) {
	cv::Vec3f original, paralell;
	equation_of_line(src0, src1, original);
	float c = - (point.x * original[0] + point.y * original[1]);
	paralell = cv::Vec3f(original[0], 1, c);

}

void
OpencvUtils::get_perpendicular_line(cv::Mat& image, cv::Point2f& src0,
		cv::Point2f& src1, cv::Point2f point,
		cv::Point2f& dst0, cv::Point2f& dst1) {

	cv::Vec3f src, perpendicular;
	equation_of_line(src0, src1, src);
	std::cout << "line: " << src << std::endl;
	std::cout << "point: " << point << std::endl;
	float m = 1 / src[0];
	std::cout << "m: " << m << std::endl,
	perpendicular = cv::Vec3f(m, 1, m * (- point.x) - point.y);
	std::cout << "perpendicular line: " << perpendicular << std::endl;

	dst0 = point;
	dst1 = cv::Point2f(- perpendicular[2] / perpendicular[0], image.cols);
}

int
OpencvUtils::draw_large_contour(cv::Mat& drawing,
        std::vector<std::vector<cv::Point>>& contours,
        const cv::Scalar& color) {

    int large_contour = get_index_large_contour(contours);

    cv::drawContours(drawing, contours, large_contour, color, 5, 8,
            std::vector<cv::Vec4i>(), 0, cv::Point());

    return large_contour;
}

void
OpencvUtils::draw_all_contours(cv::Mat& drawing,
        std::vector<std::vector<cv::Point>>& contours,
        const cv::Scalar& color) {

    for (unsigned int i = 0; i < contours.size(); i++)
        cv::drawContours(drawing, contours, i, color, 5, 8,
                std::vector<cv::Vec4i>(), 0, cv::Point());
}

void
OpencvUtils::draw_all_contours(cv::Mat& drawing,
        std::vector<std::vector<cv::Point>>& contours) {

	for (unsigned int i = 0; i < contours.size(); i++)
        cv::drawContours(drawing, contours, i,
                CV_RGB(rand() % 256, rand() % 256, rand() % 256), 5, 8,
                std::vector<cv::Vec4i>(), 0, cv::Point());
}

void
OpencvUtils::draw_borders(cv::Mat& image, cv::Scalar color) {

    int thickness = 40;

    cv::line(image, cv::Point(0, 0), cv::Point(0, image.rows), color,
            thickness);

    cv::line(image, cv::Point(image.cols, 0), cv::Point(image.cols, image.rows),
            color, thickness);

    cv::line(image, cv::Point(0, image.rows), cv::Point(image.cols, image.rows),
            color, thickness);

    cv::line(image, cv::Point(0, 0), cv::Point(image.cols, 0), color,
            thickness);
}

void
OpencvUtils::min_rotated_rect_in_large_contour(
		const std::vector<std::vector<cv::Point>>& contours,
		cv::RotatedRect& minRect) {

	int index = get_index_large_contour(contours);

    std::vector<cv::Point> points;
    for (auto& point : contours[index])
        points.push_back(point);

    minRect = cv::minAreaRect(cv::Mat(points));
}

void
OpencvUtils::min_rotated_rect(
		const std::vector<std::vector<cv::Point>>& contours,
        cv::RotatedRect& minRect) {

    std::vector<cv::Point> points;
    for (auto& vector : contours)
        for (auto& point : vector)
            points.push_back(point);

    minRect = cv::minAreaRect(cv::Mat(points));
}

template<typename Type> void
OpencvUtils::swap(Type& var1, Type& var2) {
	Type aux = var1;
	var1 = var2;
	var2 = aux;
}

template<typename Type> Type
OpencvUtils::point_between(Type p0, Type p1,
        float percentage) {

    Type pf;
    if (p0.x > p1.x) {
        pf = (p0 - p1) * (percentage / 100);
        pf += p1;

    } else {
        pf = (p1 - p0) * (percentage / 100);
        pf += p0;
    }

    return pf;
}

void
OpencvUtils::cut_rectangle2(cv::Mat& dst, cv::RotatedRect& rect,
        float percentage) {

    cv::Point2f points[4];
    rect.points(points);

    cv::Point2f p0, p1, pf0;
    p0 = point_between(points[0], points[3], percentage);
    p1 = point_between(points[1], points[2], percentage);

    pf0 = (points[0].x > points[3].x) ? points[0] : points[3];

    while (p0.x < pf0.x)
        cv::line(dst, cv::Point(p0.x++, p0.y), cv::Point(p1.x++, p1.y),
                CV_RGB(255, 255, 255), 5);
}

void
OpencvUtils::rotate(cv::Mat& src, double angle, cv::Mat& dst) {
	if (angle == 0) return;
	if (angle == 180) {
		rotate(src, 90, dst);
		rotate(dst, 90, dst);
		return;
	}
	int cols = (angle == 90 || angle == -90) ? src.rows : src.cols;
	int rows = (angle == 90 || angle == -90) ? src.cols : src.rows;

	int len = std::max(src.cols, src.rows);
	cv::Point2f pt(len/2., len/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);


    cv::Rect roi;
	if (src.cols > src.rows) {
		if (angle > 0)
			roi = cv::Rect(0, 0, cols, rows);
		else
			roi = cv::Rect(len - cols, 0, cols, rows);
	}
	else {
		if (angle < 90)
			roi = cv::Rect(0, 0, cols, rows);
		else
			roi = cv::Rect(0, len - rows, cols, rows);
	}
    cv::warpAffine(src, dst, r, cv::Size(len, len));

    cv::Mat d = dst.clone();
	cv::rectangle(d, roi, cv::Scalar(0, 0, 255), 5);
	show_image("roi", d);
	cv::waitKey();

	dst = cv::Mat(dst, roi);
}

float
OpencvUtils::calculate_height_width_ratio(const cv::Rect& rectangle) {
    return (float) rectangle.height / (float) rectangle.width;
}

float
OpencvUtils::calculate_height_width_ratio(const cv::RotatedRect& rectangle) {

    cv::Point2f p[4];
    rectangle.points(p);

    float width = calculate_distance(p[0], p[1]);
    float height = calculate_distance(p[0], p[3]);

    if (width > height)
    	swap(width, height);

    return height / width;
}

float
OpencvUtils::calculate_height_width_ratio(const cv::Mat& image) {
    return (image.cols > image.rows) ?
    		((float) image.cols / (float) image.rows) :
			((float) image.rows / (float) image.cols);
}

bool
OpencvUtils::found_color(cv::Mat& image, cv::Vec3b color, cv::Point2f& p0,
		cv::Point2f& p1) {
	bool color_found = false;

	bool axis = (p0.x == p1.x) ? true : false;
	float value = (p0.x == p1.x) ? p0.x : p0.y;

	float offset = (value > 1) ? 0.97 : 0.03;

	cv::Point2f point0, point1;
	if (axis) {
		point0 = cv::Point2f(image.cols * offset, 0);
		point1 = cv::Point2f(image.cols * offset, image.rows);
	}
	else {
		point0 = cv::Point2f(0, image.rows * offset);
		point1 = cv::Point2f(image.cols, image.rows * offset);
	}

    cv::LineIterator line_it(image, point0, point1);
    for (int i = 0; i < line_it.count && !color_found; i++, ++line_it)
        if (image.at<cv::Vec3b>(line_it.pos()) == color)
        	color_found = true;

    return color_found;
}

float
OpencvUtils::count_no_color_pixels(const cv::Mat& image, cv::Point p0,
        cv::Point p1, cv::Vec3b color) {

	float size = 0;
	cv::LineIterator line_it(image, p0, p1);

	for (int i = 0; i < line_it.count; i++, ++line_it)
		if (image.at<cv::Vec3b>(line_it.pos()) != color)
			size++;

	return size;
}

float
OpencvUtils::count_color_pixels(const cv::Mat& image, cv::Point p0,
        cv::Point p1, cv::Vec3b color) {

    float size = 0;
    cv::LineIterator line_it(image, p0, p1);

    for (int i = 0; i < line_it.count; i++, ++line_it)
        if (image.at<cv::Vec3b>(line_it.pos()) == color)
            size++;

    return size;
}

void
OpencvUtils::delete_large_contour(
        std::vector<std::vector<cv::Point>>& contours) {

    int large_contour = get_index_large_contour(contours);
    if (cv::contourArea(contours[large_contour]) < 150)
        contours.erase(contours.begin() + large_contour);
}

bool
OpencvUtils::closer_to_border(cv::Mat& image, cv::Point2f& p0, cv::Point2f& p1) {
	return (std::min(std::min(std::abs(p0.x), std::abs(p0.y)),
					std::min(image.cols - std::abs(p0.x),
							image.rows - std::abs(p0.y)))
					< std::min(std::min(std::abs(p1.x), std::abs(p1.y)),
					std::min(image.cols - std::abs(p1.x),
							image.rows - std::abs(p1.y))));
}

void
OpencvUtils::expand_to_border(cv::Mat& image, cv::Point2f& p0, cv::Point2f& p1) {
	char type = closer_to_border(image, p0, p1) &&
				(p0.x - p1.x < p0.y - p1.y) ?
				'<' : '>';
	expand_line(image, p0, p1, type);
}

void
OpencvUtils::expand_line(cv::Mat& image, cv::Point2f& p1, cv::Point2f& p2,
        char type) {

    cv::Point2f p, q;
    float m, b;

    p = cv::Point(0, 0);
    q = cv::Point(image.cols, image.rows);

    if (p1.x != p2.x) {
        p.x = 0;
        q.x = image.cols;

        m = (p1.y - p2.y) / (p1.x - p2.x);
        b = p1.y - (m * p1.x);

        p.y = m * p.x + b;
        q.y = m * q.x + b;

    } else {
        p.x = q.x = p2.x;
        p.y = 0;
        q.y = image.rows;
    }

    //cv::line(image, p, q, CV_RGB(255, 255, 255), 5);

    if (image.cols > image.rows) {
        if (type == '>') {
            if (p1.x > p2.x)
                p1 = p.x > q.x ? p : q;
            else
                p2 = p.x > q.x ? p : q;
        } else {
            if (p1.x < p2.x)
                p1 = p.x < q.x ? p : q;
            else
                p2 = p.x < q.x ? p : q;
        }

    } else {
        if (type == '>') {
            if (p1.y < p2.y)
                p1 = p.y < q.y ? p : q;
            else
                p2 = p.y < q.y ? p : q;
        } else {
            if (p1.y > p2.y)
                p1 = p.y > q.y ? p : q;
            else
                p2 = p.y > q.y ? p : q;
        }
    }
    cut_line_to_border(image, p1, p2);
}

void
OpencvUtils::cut_line_to_border(cv::Mat& image, cv::Point2f& p1, cv::Point2f& p2) {
	bool last_inside = false;

	cv::LineIterator line_it(image, p1, p2);
    cv::Point point;

    for (int i = 0; i < line_it.count; i++, ++line_it){
    	point = line_it.pos();
    	if(!last_inside && point_inside(image, point)) {
    		last_inside = true;
            p1 = point;
    	}
    	if(last_inside && point_inside(image, point))
    	    p2 = point;
    }
}

template<typename Type> bool
OpencvUtils::point_inside(cv::Mat& image, Type& p) {
	return p.x > 0 && p.x < image.cols && p.y > 0 && p.y < image.rows;
}

void
OpencvUtils::cut_rectangle(const cv::RotatedRect& rect,
		const cv::Mat& image, cv::Mat& cropped) {

	// rect is the RotatedRect (I got it from a contour...)
	// matrices we'll use
	cv::Mat M, rotated;
	// get angle and size from the bounding box
	float angle = rect.angle;
	cv::Size rect_size = rect.size;
	// thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
	if (rect.angle < -45.) {
		angle += 90.0;
		swap(rect_size.width, rect_size.height);
	}
	// get the rotation matrix
	M = getRotationMatrix2D(rect.center, angle, 1.0);
	// perform the affine transformation
	warpAffine(image, rotated, M, image.size(), cv::INTER_CUBIC);
	// crop the resulting image
	getRectSubPix(rotated, rect_size, rect.center, cropped);
}

void
OpencvUtils::equation_of_line(cv::Point2f& p0, cv::Point2f& p1,
		cv::Vec3f& equation) {
	// In the way Ax + By + C = 0 => Vec3f(A, B, C)

	if ((p1.x - p0.x) != 0){
		float m = (p1.y - p0.y) / (p1.x - p0.x);
		float offset = m * p0.x - p0.y;
		//Vec3f: x * m + y * 1 + offset = 0
		equation = cv::Vec3f(-m, 1, offset);

	} else
		equation = cv::Vec3f(1, 0, - p0.x);
}

void
OpencvUtils::bisector_two_lines(cv::Point2f& p0, cv::Point2f& p1,
		cv::Point2f& p2, cv::Point2f& p3, cv::Vec3f& equation_result,
		bool type) {
	cv::Vec3f equation0, equation1;
	equation_of_line(p0, p1, equation0);
	equation_of_line(p2, p3, equation1);

	bisector_two_lines(equation0, equation1, equation_result, type);
}

void
OpencvUtils::bisector_two_lines(cv::Vec3f& equation0, cv::Vec3f& equation1,
		cv::Vec3f& equation_result, bool type) {
	float division0 = sqrt(equation0[0] * equation0[0] +
			equation0[1] * equation0[1]);
	float division1 = sqrt(equation1[0] * equation1[0] +
			equation1[1] * equation1[1]);

	cv::Vec3f aux0(equation0[0] * division1, equation0[1] * division1,
			equation0[2] * division1);
	cv::Vec3f aux1(equation1[0] * division0, equation1[1] * division0,
			equation1[2] * division0);

	equation_result = type ?
			cv::Vec3f(aux0[0] + aux1[0], aux0[1] + aux1[1],
			aux0[2] + aux1[2]) :
			cv::Vec3f(aux0[0] - aux1[0], aux0[1] - aux1[1],
			aux0[2] - aux1[2]);
}

void
OpencvUtils::intersection_of_bisectors_inside_image(cv::Mat& image,
		cv::Vec3f& equation0, cv::Vec3f& equation1,
		cv::Vec3f& equation2, cv::Point2f& intersection,
		float& radius) {

	cv::Vec3f bisectors[4];
	bisector_two_lines(equation0, equation1, bisectors[0], true);
	bisector_two_lines(equation0, equation1, bisectors[1], false);
	bisector_two_lines(equation1, equation2, bisectors[2], true);
	bisector_two_lines(equation1, equation2, bisectors[3], false);

	cv::Point2f intersections[4];
	intersection_of_lines(bisectors[0], bisectors[2], intersections[0]);
	intersection_of_lines(bisectors[0], bisectors[3], intersections[1]);
	intersection_of_lines(bisectors[1], bisectors[2], intersections[2]);
	intersection_of_lines(bisectors[1], bisectors[3], intersections[3]);

	int i;
	for(i = 0; i < 4; i++)
		if(intersections[i].x > 0 &&  intersections[i].x < image.cols
				&& intersections[i].y > 0 && intersections[i].y < image.rows)
			break;

	intersection = intersections[i];
	radius = abs(distance_from_point_to_rect(intersection, equation1));
}

void
OpencvUtils::intersection_of_bisectors_inside_image(cv::Mat& image,
		cv::Point2f& p0, cv::Point2f& p1,
		cv::Point2f& p2, cv::Point2f& p3,
		cv::Point2f& intersection, float& radius) {

	cv::Vec3f equation0, equation1, equation2;
	equation_of_line(p0, p1, equation0);
	equation_of_line(p1, p2, equation1);
	equation_of_line(p2, p3, equation2);

	intersection_of_bisectors_inside_image(image, equation0, equation1,
			equation2, intersection, radius);

}

float
OpencvUtils::distance_from_point_to_rect(cv::Point2f& point, cv::Vec3f& rect) {
	return (rect[0] * point.x + rect[1] * point.y + rect[2]) /
			sqrt(rect[0] * rect[0] + rect[1] * rect[1]);
}

void
OpencvUtils::point_minimal_distance_from_point_to_rect(cv::Point2f& point,
		cv::Vec3f& rect, cv::Point2f& solution) {
	float x = (rect[1] * (rect[1] * point.x - rect[0] * point.y)
			- rect[0] * rect[2]) / (rect[0] * rect[0] + rect[1] * rect[1]);

	float y = (rect[0] * (- rect[1] * point.x + rect[0] * point.y)
			- rect[1] * rect[2]) / (rect[0] * rect[0] + rect[1] * rect[1]);
	solution = cv::Point2f(x, y);
}

void
OpencvUtils::intersection_of_lines(cv::Vec3f& equation0,
		cv::Vec3f& equation1, cv::Point2f& intersection_point) {

	cv::Vec3f aux0(equation0[0] / equation0[0], equation0[1] / equation0[0],
			equation0[2] / equation0[0]);
	cv::Vec3f aux1(equation1[0] / equation1[0], equation1[1]  / equation1[0],
			equation1[2] / equation1[0]);

	cv::Vec3f res_partial = cv::Vec3f(aux0[0] - aux1[0], aux0[1] - aux1[1],
			 aux0[2] - aux1[2]);

	float res_x, res_y;
	res_y = - res_partial[2] / res_partial[1];
	res_x = - aux0[2] + (- aux0[1]) * res_y;

	intersection_point = cv::Point2f(res_x, res_y);
}

void
OpencvUtils::get_rois(cv::Mat& image, int number_of_slices,
		std::vector<std::vector<cv::Rect>>& rois) {
	int cuts = number_of_slices;

	std::vector<cv::Rect> aux;
	int offset_x = image.cols / cuts;
	int offset_y = image.rows / cuts;
	std::cout << "rows " << image.rows << std::endl;
	std::cout << "cols " << image.cols << std::endl;

	for (int i = 0; i < cuts; i++) {
		aux.clear();
		for (int j = 0; j < cuts; j++)
			aux.push_back(cv::Rect(offset_x * i, offset_y * j, offset_x, offset_y));

		rois.push_back(aux);
	}
}

void
OpencvUtils::cut_image(cv::Mat& image, int number_of_slices,
		std::vector<std::vector<cv::Mat>>& slices) {
	std::vector<std::vector<cv::Rect>> rois;
	cv::Mat image_roi;
	std::vector<cv::Mat> aux;

	get_rois(image, number_of_slices, rois);

	for (auto& row: rois) {
		aux.clear();
		for (auto& roi: row) {
			image_roi = cv::Mat(image, roi);
			aux.push_back(image_roi);
		}
		slices.push_back(aux);
	}
}

void
OpencvUtils::join_image(std::vector<std::vector<cv::Mat>>& slices,
		cv::Mat& image) {
	cv::Mat row_image;
	for (int i = 0; i < (int) slices.size(); i++) {
		for (int j = 0; j < (int) slices[i].size(); j++) {
			if (j == 0) row_image = slices[j][i];
			else
				cv::hconcat(row_image, slices[j][i], row_image);
		}
		if (i == 0) image = row_image;
		else
			cv::vconcat(image, row_image, image);
	}
}

float
OpencvUtils::get_variance(cv::Mat& image) {

	float mean = cv::mean(image)[0];
	std::cout << "mean: " << mean << std::endl;
	float cumulative = 0, value = 0;
	for(int i = 0; i < image.rows; i++)
		for(int j = 0; j < image.cols; j++){
			value = image.at<cv::Vec3b>(i, j)[0] - mean;
			cumulative += (value * value);
		}
	return cumulative / (image.rows * image.cols);
}

void
OpencvUtils::adaptative_threshold_on_sliced_image(cv::Mat& image,
		int number_of_slices, int max_variance) {
	std::vector<std::vector<cv::Mat>> original_slices, thesh_slices;
	std::vector<cv::Mat> vector_slices;
	cv::Mat aux;

	cut_image(image, number_of_slices, original_slices);

	for (auto& row: original_slices) {
		vector_slices.clear();
		for (auto& slice: row) {
			aux = cv::Mat();
			std::cout << "variance: " << sqrt(get_variance(slice)) << std::endl;
			if(sqrt(get_variance(slice)) > max_variance)
				cv::threshold(slice, aux, 0,
					255, cv::THRESH_BINARY + cv::THRESH_OTSU);
			else
				aux = cv::Mat(slice.rows, slice.cols,
						slice.type(), cv::Scalar(255, 255, 255));
			vector_slices.push_back(aux);
		}
		thesh_slices.push_back(vector_slices);
	}
	join_image(thesh_slices, image);
}


// From here not compile on windows OS with Visual Studio 2013 compiler
void
OpencvUtils::trackbar_find_contours(CallbackData& data) {

    cv::TrackbarCallback callback = callback_threshold;
    std::string name = "find_contours_trackbar";

    cv::namedWindow(name, cv::WINDOW_NORMAL);

    char *trackbarName;
    trackbarName = "type:\n" \
			" 0: CV_RETR_EXTERNAL\n" \
			" 1: CV_RETR_LIST\n" \
			" 2: CV_RETR_CCOMP\n";

    cv::createTrackbar(trackbarName,
    					name,
    					&data.slider1,
    					3,
    					callback,
    					&data);

    trackbarName = "type:\n" \
			" 0: CV_CHAIN_APPROX_NONE\n" \
			" 1: CV_CHAIN_APPROX_SIMPLE\n" \
			" 2: CV_CHAIN_APPROX_TC89_L1\n" \
			" 3: CV_CHAIN_APPROX_TC89_KCOS\n";

    cv::createTrackbar(trackbarName,
    					name,
    					&data.slider2,
    					4,
    					callback,
    					&data);

    callback(0, &data);

    cv::imshow(name, data.result);
    cv::waitKey();

    std::cout << "Method: find_contours" << "\nSlider1 value: "
    		 << data.slider1 << "\nSlider2 value: " << data.slider2 << std::endl;
}

void
OpencvUtils::trackbar_adaptative_threshold(CallbackData& data) {

    cv::TrackbarCallback callback = callback_adaptative_threshold;
    std::string name = "adaptative_threshold_trackbar";

    cv::namedWindow(name, cv::WINDOW_NORMAL);

    char *trackbarName = "Background value:";
	cv::createTrackbar(trackbarName,
						name,
						&data.slider1,
						255,
						callback,
						&data);

	trackbarName = "Block size:";
	cv::createTrackbar(trackbarName,
						name,
						&data.slider2,
						255,
						callback,
						&data);

    if (data.type == -1){
        data.type = 0;
        trackbarName = "type:\n" \
                " 0: ADAPTIVE_THRESH_MEAN_C\n" \
                " 1: ADAPTIVE_THRESH_GAUSSIAN_C\n";

        cv::createTrackbar(trackbarName,
                            name,
                            &data.type,
                            2,
                            callback,
                            &data);
        data.type2 = 0;
        trackbarName = "type:\n" \
                " 0: THRESH_BINARY\n" \
                " 1: THRESH_BINARY_INV\n";

        cv::createTrackbar(trackbarName,
                            name,
                            &data.type2,
                            2,
                            callback,
                            &data);
    }

    callback(0, &data);

    cv::imshow(name, data.result);
    cv::waitKey();

    std::cout << "Method: adaptative_threshold" << "\nSlider1 value: "
    		 << data.slider1 << "\nSlider2 value: " << data.slider2 << std::endl;

    std::cout << "Type value: " << data.type << std::endl;
}

void
OpencvUtils::trackbar_canny(CallbackData& data) {
    cv::TrackbarCallback callback = callback_canny;
    std::string name = "canny_trackbar";

    cv::namedWindow(name, cv::WINDOW_NORMAL);

    char *trackbarName = "threshold value 1:";
	cv::createTrackbar(trackbarName,
						name,
						&data.slider1,
						255,
						callback,
						&data);

	trackbarName = "threshold value 2:";
	cv::createTrackbar(trackbarName,
						name,
						&data.slider2,
						255,
						callback,
						&data);

	callback(0, &data);

	cv::imshow(name, data.result);
	cv::waitKey();

	std::cout << "Method: canny" << "\nSlider1 value: " << data.slider1
	            << "\nSlider2 value: " << data.slider2 << std::endl;
}

void
OpencvUtils::trackbar_threshold(CallbackData& data) {

    cv::TrackbarCallback callback = callback_threshold;
    std::string name = "threshold_trackbar";

    cv::namedWindow(name, cv::WINDOW_NORMAL);

    char *trackbarName = "threshold value:";
	cv::createTrackbar(trackbarName,
						name,
						&data.slider1,
						255,
						callback,
						&data);

	trackbarName = "background value:";
	cv::createTrackbar(trackbarName,
						name,
						&data.slider2,
						255,
						callback,
						&data);

    if (data.type == -1) {
        data.type = 0;
        trackbarName = "type:\n" \
                " 0: Binary\n" \
                " 1: Binary Inverted\n" \
                " 2: Threshold Truncated\n" \
                " 3: Threshold to Zero\n" \
                " 4: Threshold to Zero Inverted\n"\
				" 5: Binary + OTSU\n" \
				" 6: Binary Inverted + OTSU\n" \
				" 7: Threshold Truncated + OTSU\n" \
				" 8: Threshold to Zero + OTSU\n" \
				" 9: Threshold to Zero Inverted + OTSU";
        cv::createTrackbar(trackbarName,
                            name,
                            &data.type,
                            9,
                            callback,
                            &data);

    }

    callback(0, &data);

    cv::imshow(name, data.result);
    cv::waitKey();

    std::cout << "Method: threshold" << "\nSlider1 value: " << data.slider1
            << "\nSlider2 value: " << data.slider2 << std::endl;

    std::cout << "Type value: " << data.type << std::endl;
}

void
OpencvUtils::create_trackbar(cv::Mat& src, cv::Mat& dst, std::string method, int type) {

    std::map<std::string, std::function<void(CallbackData&)>> callbacks;
    callbacks = {
            {"threshold", 			 OpencvUtils::trackbar_threshold},
			{"adaptative_threshold", OpencvUtils::trackbar_adaptative_threshold},
            {"canny",     			 OpencvUtils::trackbar_canny},
			{"find_contours",		 OpencvUtils::trackbar_find_contours},
    };

    CallbackData data;
    data.image = src.clone();
    data.result = src.clone();
    data.type = type;

    callbacks[method](data);

    dst = data.result.clone();
}

void
OpencvUtils::callback_canny(int value, void *ptr) {

    CallbackData& data = *((CallbackData*) (ptr));

    cv::Canny(data.image, data.result, data.slider1, data.slider2);

    show_image("canny_trackbar", data.result);
}

void
OpencvUtils::callback_threshold(int value, void *ptr) {

    CallbackData& data = *((CallbackData*) (ptr));
    if (data.type >= 5) data.type += 3;
    cv::threshold(data.image, data.result, data.slider1,
            data.slider2, data.type);
    if (data.type >= 5) data.type -= 3;

    show_image("threshold_trackbar", data.result);
}

void
OpencvUtils::callback_adaptative_threshold(int value, void *ptr) {

    CallbackData& data = *((CallbackData*) (ptr));

    cv::adaptiveThreshold(data.image, data.result, data.slider1,
    		data.type, data.type2, data.slider2, 5);

    show_image("adaptative_threshold_trackbar", data.result);

}

void
OpencvUtils::callback_find_contours(int value, void *ptr) {
    CallbackData& data = *((CallbackData*) (ptr));

    std::vector<std::vector<cv::Point>> contours;

    data.slider2++;
    cv::findContours(data.image, contours, data.slider1,
            data.slider2);
    data.slider2--;
    OpencvUtils::draw_all_contours(data.result, contours);
    show_image("find_contours_trackbar", data.result);

}

