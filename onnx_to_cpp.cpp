#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main(int argc, char* argv[]) {
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	Ort::SessionOptions session_options;
	session_options.SetThreadPoolSize(1);

	session_options.SetGraphOptimizationLevel(1);

#ifdef _WIN32
	const wchar_t* model_path = L"R2plus1D_model.onnx";
#else
	const char* model_path = "R2plus1D_model.onnx";
#endif

	printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_options);

	Ort::Allocator allocator = Ort::Allocator::CreateDefault();

	size_t num_input_nodes = session.GetInputCount();
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;

	// iterate over all input nodes
	for (int i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name = session.GetInputName(i, allocator);
		input_node_names[i] = input_name;

		// print input node types
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		// print input shapes/dims
		input_node_dims = tensor_info.GetShape();
	}

	size_t input_tensor_size = 1 * 3 * 16 * 224 * 224;

	std::vector<float> input_tensor_values(input_tensor_size);
	std::vector<const char*> output_node_names = { "r2plus1d_flatten0_reshape0" };

	for (unsigned int i = 0; i < input_tensor_size; i++)
		input_tensor_values[i] = (float)i / (input_tensor_size + 1);

	Ort::AllocatorInfo allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());

	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

	float* floatarr = output_tensors.front().GetTensorMutableData<float>();

	for (int i = 0; i < 5; i++)
		printf("Score for class [%d] =  %f\n", i, floatarr[i]);

	printf("Done!\n");
	return 0;
}
