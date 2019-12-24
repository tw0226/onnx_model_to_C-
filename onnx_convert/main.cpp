#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "providers/cuda/cuda_provider_factory.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>
#include <time.h>
#include "action_infer.h"

int main(void) {
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	Ort::SessionOptions session_options;
	int deviceID = 0;
	session_options.SetThreadPoolSize(1);
	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, deviceID);
	session_options.SetGraphOptimizationLevel(1);
	std::vector<cv::Mat> imgs;
	std::mutex m;
	const wchar_t* model_name = L"R2plus1D_model_v3.onnx";
	Action_infer infer(model_name, env, deviceID, 1);
	infer.SetInputOutputSet();
	cv::VideoCapture cap("C026100_0021.mp4");
	std::ifstream in("C026100_0021.txt");
	std::string f_n, str_state, str_x, str_y, str_w, str_h;
	int width = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
	int height = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);

	
	
	cv::Point point(20, 20);
	int count = 0;
	while (cap.isOpened()) {
		cv::Mat frame;
		cap >> frame;
		std::string state = "None";
		/// 사람 중심 ROI Crop 하는 code 부분
		in >> f_n >> str_state >> str_x >> str_y >> str_w >> str_h;
		if (!in.is_open() || frame.empty())
			break;
		int x = std::stoi(str_x);
		int y = std::stoi(str_y);
		int w = std::stoi(str_w);
		int h = std::stoi(str_h);
		//std::cout << f_n << " " << str_state << " " << x << " " << y << " " << w << " " << h << std::endl;
	
		if (x + w > width) {
			w = width - x;
		}
		if (y + h > height) {
			h = height - y;
		}
		if (in.is_open()) {
			cv::Rect rect(x, y, w, h);
			frame = frame(rect);
		}
		/// End
		count++;
		infer.video_input(frame);
		cv::resize(frame, frame, cv::Size(224, 224));
		
		if (count < 16)
			continue;
		else {
			float output = infer.GetOutput(m);
			if (output > 0.5)
				state = "Fall_Down";
			else
				state = "None";
		}

		cv::putText(frame, state, point, 1, 1.2, cv::Scalar(255, 0, 0), 2);
		cv::imshow("asdf", frame);
		cv::waitKey(1);
	}

	return 0;
}