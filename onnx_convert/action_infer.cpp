#include "action_infer.h"
#include <onnxruntime_cxx_api.h>
#include "providers/cuda/cuda_provider_factory.h"
#include <chrono>
#include <iostream>
#include <assert.h>
#include <mutex>
#include <cuda.h>
#include <cuda_runtime_api.h>
#ifdef _WIN32

Action_infer::Action_infer(const wchar_t* model_path, Ort::Env &env, int deviceID, int batch)
{
	this->batch = batch;
	std::cout << "model_path : " << model_path << std::endl;
	this->deviceid = deviceID;
	Ort::SessionOptions session_options;
	session_options.SetThreadPoolSize(1);
	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, deviceID);
	session_options.SetGraphOptimizationLevel(1);	
	session = std::shared_ptr<Ort::Session>(new Ort::Session(env, model_path, session_options));
	std::cout << "Model Read Success" << std::endl;
}
#else
Action_infer::Action_infer(const char* model_path, Ort::Env &env, int deviceID, int batch)
{
	this->batch = batch;
	std::cout << "model_path : " << model_path << std::endl;
	this->deviceid = deviceID;
	Ort::SessionOptions session_options;
	session_options.SetThreadPoolSize(1);

	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, deviceID);
	session_options.SetGraphOptimizationLevel(1);

	session = std::shared_ptr<Ort::Session>(new Ort::Session(env, model_path, session_options));

	std::cout << "Model Read Success" << std::endl;
}
#endif

void Action_infer::SetInputOutputSet() {
	Ort::Allocator allocator = Ort::Allocator::CreateDefault();
	std::cout << "session->GetInputCount()" << session->GetInputCount() << std::endl;
	std::cout << "session->GetOutputCount()" << session->GetOutputCount() << std::endl;
	std::cout << "session->GetInputName(0, allocator)" << session->GetInputName(0, allocator) << std::endl;
#ifdef _DEBUG
	for (int i = 0; i < session->GetInputCount(); i++) {
		input_node_names.push_back(session->GetInputName(i, allocator));
		input_node_dims = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
	}	
	output_node_names = { "output1" };
#else
	input_node_names.reserve(session->GetInputCount());
	output_node_names.reserve(session->GetOutputCount());
	input_node_names[0] = session->GetInputName(0, allocator);
	input_node_dims = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	char* output_name = session->GetOutputName(0, allocator);
	for (int i = 0; i < session->GetOutputCount(); ++i)
		output_node_names[i] = session->GetOutputName(i, allocator);
#endif

}

float Action_infer::GetOutput(std::mutex &m) {
	std::vector<float> input_vector = Vector_change(imgs);
	Ort::AllocatorInfo allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_vector.data(), input_vector.size(), input_node_dims.data(), input_node_dims.size());
	assert(input_tensor.IsTensor());
	m.lock();
	cudaSetDevice(this->deviceid);
	
	auto output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	m.unlock();
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	float output = 1 / (1 + std::exp(-*floatarr));
	std::string state;
	if (output > 0.5)
		state = "Fall_Down";
	else
		state = "None";
	printf("Score for class =  %f %f\n", *floatarr, output);
	input_tensor_values.clear();
	return output;
}
void Action_infer::video_input(cv::Mat frame) {
	frame.convertTo(frame, CV_32FC3);
	frame /= (float)255.0;
	cv::resize(frame, frame, cv::Size(224, 224));
	if (imgs.size() < 16)
		imgs.push_back(frame);
	else {
		imgs.push_back(frame);
		imgs.erase(imgs.begin(), imgs.begin() + 1);
	}
}

std::vector<float> Action_infer::Vector_change(std::vector<cv::Mat> imgs) {
	for (int i = 0; i < frame_length; i++) {
		for (int j = 0; j < imgs[i].rows; j++) {
			input_tensor_values.insert(input_tensor_values.end(), imgs[i].ptr<float>(j), imgs[i].ptr<float>(j) + imgs[i].cols * 3);
		}
	}
	return input_tensor_values;
}
