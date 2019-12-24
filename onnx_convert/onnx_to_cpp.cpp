//#include <assert.h>
//#include <vector>
//#include <onnxruntime_cxx_api.h>
//#include "providers/cuda/cuda_provider_factory.h"
//#include "opencv/highgui.h"
//#include "opencv2/opencv.hpp"
//#include <iostream>
//#include <chrono>
//#include <time.h>
//#include "action_infer.h"
//
///*
//namespace my_action_onnx {
//	onnx_action_module::onnx_action_module() {
//		
//	}
//	void onnx_action_module::onnx_init(const wchar_t *model_name) {
//		env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
//		session_options.SetThreadPoolSize(1);
//		OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
//		session_options.SetGraphOptimizationLevel(1);
//		session = Ort::Session(env, model_name, session_options);
//		allocator = Ort::Allocator::CreateDefault();
//		size_t num_input_nodes = session.GetInputCount();
//		std::vector<const char*> input_node_names(num_input_nodes);
//		input_node_names.reserve(session.GetInputCount());
//		std::vector<int64_t> input_node_dims;
//		for (int i = 0; i < num_input_nodes; i++) {
//			// print input node names
//			char* input_name = session.GetInputName(i, allocator);
//			input_node_names[i] = input_name;
//
//			// print input node types
//			Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
//			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
//
//			// print input shapes/dims
//			input_node_dims = tensor_info.GetShape();
//		}
//		//allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//	}
//	void onnx_action_module::set_input_tensor_size(size_t size) {
//		input_tensor_size = size;
//	}
//	void onnx_action_module::set_VideoCapture(std::string filename) {
//		cap = cv::VideoCapture(filename);
//		cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
//		cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
//	}
//	cv::VideoCapture onnx_action_module::get_Video() {
//		return cap;
//	}
//	void onnx_action_module::set_copy_mat(cv::Mat mat) {
//		copy_mat = mat;
//	}
//	cv::Mat onnx_action_module::get_copyMat() {
//		return copy_mat;
//	}
//	void onnx_action_module::push_imgs(cv::Mat mat) {
//		if (imgs.size() < 16)
//			imgs.push_back(mat);
//		else {
//			imgs.push_back(mat);
//			imgs.erase(imgs.begin(), imgs.begin() + 1);
//		}
//	}
//	std::vector<cv::Mat> onnx_action_module::get_imgs() {
//		return imgs;
//	}
//	void onnx_action_module::vector_change() {
//		for (int i = 0; i < frame_length; i++) {
//			for (int j = 0; j < imgs[i].rows; j++) {
//				input_tensor_values.insert(input_tensor_values.end(), imgs[i].ptr<float>(j), imgs[i].ptr<float>(j) + imgs[i].cols * 3);
//			}
//		}
//	}
//	std::vector<float> onnx_action_module::get_input_tensor_values() {
//		return input_tensor_values;
//	}
//	Ort::AllocatorInfo onnx_action_module::get_AllocatorInfo() {
//		return Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//	}
//	Ort::Value onnx_action_module::get_input_tensor() {
//		return Ort::Value::CreateTensor<float>(allocator_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 5);
//	}
//	std::vector<const char*> onnx_action_module::get_input_node_names() {
//		return input_node_names;
//	}
//	Ort::Session onnx_action_module::get_Session() {
//		return session;
//	}
//
//	std::vector<int64_t> onnx_action_module::get_input_node_dims() {
//		return input_node_dims;
//	}
//	
//}
//*/
//
//
//using namespace std::chrono;
//const int frame_length = 16;
//std::vector<float> Vector_change(std::vector<cv::Mat> input);
//
//int main(int argc, char* argv[]) {
//
//	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
//	Ort::SessionOptions session_options;
//	session_options.SetThreadPoolSize(1);
//	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
//	session_options.SetGraphOptimizationLevel(1);
//
//#ifdef _WIN32
//	//const wchar_t* model_name = L"squeezenet1.1.onnx";
//	const wchar_t* model_name = L"R2plus1D_model_v3.onnx";
//#else
//	const char* model_name = "R2plus1D_model.onnx";
//#endif
//
//	Ort::Session session(env, model_name, session_options);
//
//	Ort::Allocator allocator = Ort::Allocator::CreateDefault();
//	size_t num_input_nodes = session.GetInputCount();
//	std::vector<const char*> input_node_names(num_input_nodes);
//	input_node_names.reserve(session.GetInputCount());
//	std::vector<int64_t> input_node_dims;
//
//	// iterate over all input nodes
//	for (int i = 0; i < num_input_nodes; i++) {
//		// print input node names
//		char* input_name = session.GetInputName(i, allocator);
//		input_node_names[i] = input_name;
//
//		// print input node types
//		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
//		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
//
//		// print input shapes/dims
//		input_node_dims = tensor_info.GetShape();
//	}
//
//	size_t input_tensor_size = 1 * 3 * 16 * 224 * 224;
//	
//	//my_action_onnx::onnx_action_module action;
//	
//#ifdef _WIN32
//	const wchar_t* model_name = L"R2plus1D_model_v3.onnx";
//#else
//	const char* model_name = "R2plus1D_model.onnx";
//#endif
//
//	action.onnx_init(model_name);
//	size_t input_tensor_size = 1 * 3 * 16 * 224 * 224;
//	action.set_input_tensor_size(input_tensor_size);
//	action.set_VideoCapture("C026100_0021.mp4");
//	std::vector<cv::Mat> imgs;
//	cv::Point point(20, 20);
//	while (action.get_Video().isOpened()) {
//		cv::Mat frame;
//		cv::String state;
//		action.get_Video() >> frame;
//		frame.convertTo(frame, CV_32FC3);
//		frame /= (float)255.0;
//		frame.copyTo(action.get_copyMat());
//		action.push_imgs(frame);
//		if (action.get_imgs().size() == frame_length) {
//			action.vector_change();
//			action.get_input_tensor_values();
//			std::vector<const char*> output_node_names = { "output1" };
//			
//			Ort::Value input_tensor = action.get_input_tensor();
//			assert(input_tensor.IsTensor());
//			auto output_tensors = action.get_Session().Run(Ort::RunOptions{ nullptr },
//				action.get_input_node_names.data(),
//				&input_tensor, 1,
//				output_node_names.data(), 1);
//			assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
//			float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//			float output = 1 / (1 + std::exp(-*floatarr));
//			if (output > 0.5)
//				state = "Fall_Down";
//			else
//				state = "None";
//		}
//		
//		//printf("Score for class =  %f %f\n", *floatarr, output);
//		
//		cv::putText(action.get_copyMat(), state, point, 1, 1.2, cv::Scalar(255, 0, 0), 2);
//		cv::imshow("a1", action.get_copyMat());
//		cv::waitKey(1);
//		action.get_input_tensor_values().clear();
//	}
//	action.get_Video().release();
//	/*
//	cv::VideoCapture cap("C026100_0021.mp4");
//	std::ifstream in("C026100_0021.txt");
//	std::string f_n, str_state, str_x, str_y, str_w, str_h;
//	int width = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
//	int height = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);
//	std::cout << width << " " << height << std::endl;
//	
//	std::vector<cv::Mat> imgs;
//	*/
//	
//	
//	while (cap.isOpened()) {
//		cv::Mat frame;
//		cap >> frame;
//		frame.convertTo(frame, CV_32FC3);
//		frame /= (float)255.0;
//		
//		std::string state;
//		in >> f_n >> str_state >> str_x >> str_y >> str_w >> str_h;
//		
//		//if (!in.is_open() || frame.empty())
//		//	break;
//		//int x = std::stoi(str_x);
//		//int y = std::stoi(str_y);
//		//int w = std::stoi(str_w);
//		//int h = std::stoi(str_h);
//		//std::cout << f_n << " " << str_state << " " << x << " " << y << " " << w << " " << h << std::endl;
//		//if (x + w > width) {
//		//	w = width - x;
//		//}
//		//if (y + h > height) {
//		//	h = height - y;
//		//}
//		//if (in.is_open()) {
//		//	cv::Rect rect(x, y, w, h);
//		//	frame = frame(rect);
//		//}
//		
//		cv::resize(frame, frame, cv::Size(224, 224));
//		cv::Mat clone;
//		frame.copyTo(clone);
//		if(imgs.size() < 16)
//			imgs.push_back(frame);
//		else {
//			imgs.push_back(frame);
//			imgs.erase(imgs.begin(), imgs.begin()+1);
//			clock_t start, end;
//			start = clock();
//			std::vector<float> input_tensor_values = Vector_change(imgs);
//			end = clock();
//			
//			double duration = double(end - start);
//			std::cout << "vector change time : " << duration << "ms" << std::endl;
//
//			start = clock();
//			std::vector<const char*> output_node_names = { "output1" };
//			Ort::AllocatorInfo allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//			Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 5);
//			assert(input_tensor.IsTensor());
//			auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
//			assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
//			end = clock();
//			duration = double(end - start);
//			std::cout << "onnx Inference Time : " << duration << "ms" << std::endl;
//			float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//			float output = 1 / (1 + std::exp(-*floatarr));
//			
//			if (output > 0.5)
//				state = "Fall_Down";
//			else
//				state = "None";
//			
//			//printf("Score for class =  %f %f\n", *floatarr, output);
//		}
//		cv::putText(clone, state, point, 1, 1.2, cv::Scalar(255, 0, 0), 2);
//		cv::imshow("a1", clone);
//		cv::waitKey(1);
//	}
//	in.close();
//	cap.release();
//	
//	return 0;
//	
//}
//
//std::vector<float> Vector_change(std::vector<cv::Mat> imgs) {
//	std::vector<float> input_tensor_values;
//	for (int i = 0; i < frame_length; i++) {
//		for (int j = 0; j < imgs[i].rows; j++) {
//			input_tensor_values.insert(input_tensor_values.end(), imgs[i].ptr<float>(j), imgs[i].ptr<float>(j) + imgs[i].cols * 3);
//		}
//	}
//	return input_tensor_values; 
//}
