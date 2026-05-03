// initialization.inl
#include <cmath>
#include <random>

namespace MLCore::Init {
	template <typename T>
	void Init(TensorCore::Tensor<T>& tensor, size_t fan_in, size_t fan_out, InitType type) {
		std::random_device rd;
		std::mt19937 gen(rd());

		switch (type) {
		case InitType::Zero:
			{
				tensor.Fill(static_cast<T>(0));
				break;
			}

		case InitType::XavierUniform:
			{
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
				size_t size = tensor.NumElements();

				T stddev = std::sqrt(static_cast<T>(2.0) / fan_in);
				std::normal_distribution<T> dist(0.0, stddev);

				for (size_t i = 0; i < size; ++i) {
					tensor[i] = dist(gen);
				}

				break;
			}
		}
	}
}