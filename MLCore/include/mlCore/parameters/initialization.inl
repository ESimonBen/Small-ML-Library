 /// initialization.inl
#include <cmath>
#include <mlCore/runtime/context.h>

namespace MLCore::NN {
	template <typename T>
	void Init(TensorCore::Tensor<T>& tensor, size_t fan_in, size_t fan_out, InitType type) {
		std::mt19937 gen = Runtime::MLContext::GetRNG();

		switch (type) {
		case InitType::Zero:
			{
				tensor.Fill(static_cast<T>(0));
				break;
			}

		case InitType::XavierUniform:
			{
				if (fan_in == 0) {
					throw std::runtime_error("ERROR: Init: fan_in cannot be 0");
				}

				if (fan_out == 0) {
					throw std::runtime_error("ERROR: Init: fan_out cannot be 0");
				}

				size_t size = tensor.NumElements();

				T limit = std::sqrt(static_cast<T>(6.0) / (fan_in + fan_out));
				std::uniform_real_distribution<T> dist(-limit, limit);

				for (size_t i = 0; i < size; ++i) {
					tensor[i] = dist(gen);
				}

				break;
			}

		case InitType::XavierNormal:
			{
				if (fan_in == 0) {
					throw std::runtime_error("ERROR: Init: fan_in cannot be 0");
				}

				if (fan_out == 0) {
					throw std::runtime_error("ERROR: Init: fan_out cannot be 0");
				}

				size_t size = tensor.NumElements();

				T stddev = std::sqrt(static_cast<T>(2.0) / (fan_in + fan_out));
				std::normal_distribution<T> dist(0.0, stddev);

				for (size_t i = 0; i < size; ++i) {
					tensor[i] = dist(gen);
				}

				break;
			}

		case InitType::HeUniform:
			{
				if (fan_in == 0) {
					throw std::runtime_error("ERROR: Init: fan_in cannot be 0");
				}

				size_t size = tensor.NumElements();

				T limit = std::sqrt(static_cast<T>(6.0) / fan_in);
				std::uniform_real_distribution<T> dist(-limit, limit);

				for (size_t i = 0; i < size; ++i) {
					tensor[i] = dist(gen);
				}

				break;
			}

		case InitType::HeNormal:
			{
				if (fan_in == 0) {
					throw std::runtime_error("ERROR: Init: fan_in cannot be 0");
				}

				size_t size = tensor.NumElements();

				T stddev = std::sqrt(static_cast<T>(2.0) / fan_in);
				std::normal_distribution<T> dist(0.0, stddev);

				for (size_t i = 0; i < size; ++i) {
					tensor[i] = dist(gen);
				}

				break;
			}

		default:
			throw std::runtime_error("ERROR: Init: Unknown initialization type");
		}
	}
}