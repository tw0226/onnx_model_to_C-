#pragma once
#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "providers/cuda/cuda_provider_factory.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>
#include <time.h>
#include <mutex>

class Action_infer {
private:
	std::shared_ptr<Ort::Session> session;
	std::vector<cv::Mat> imgs;
	std::vector<float> input_tensor_values;
	std::vector<int64_t> input_node_dims;
	std::vector<const char*> input_node_names;	
	std::vector<const char*> output_node_names;

	int width = 224;
	int height = 224;
	int frame_length = 16;
	int batch = -1;
	int deviceid = -1;

public:
#ifdef _WIN32
	Action_infer(const wchar_t* model_path, Ort::Env &env, int deviceID = 0, int batch = 16);
#else
	Action_infer(const char * model_path, Ort::Env &env, int deviceID = 0, int batch = 16);
#endif
	void video_input(cv::Mat frame);
	void SetInputOutputSet();
	float GetOutput(std::mutex &m);
	std::vector<float> Vector_change(std::vector<cv::Mat> imgs);
};