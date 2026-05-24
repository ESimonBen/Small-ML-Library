#include "trainer.h"
// trainer.inl

namespace MLCore::Training {
	template <typename T>
	Trainer<T>::Trainer(NN::Module<T>& model, Optimizers::Optimizer<T>& optimizer, LossFn<T> lossFn)
		: m_Model(model), m_Optimizer(optimizer), m_LossFn(lossFn)
	{}

	template <typename T>
	void Trainer<T>::Fit(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, int epochs, size_t batchSize) {
		m_Model.Train();

		if (inputs.Dims()[0] != targets.Dims()[0]) {
			throw std::runtime_error(
				"Input/target sample mismatch"
			);
		}

		if (batchSize == 0) {
			throw std::runtime_error(
				"Batch size cannot be zero"
			);
		}

		size_t numSamples = inputs.Dims()[0];

		for (int epoch = 0; epoch < epochs; ++epoch) {
			T epochLoss = 0;
			size_t batchCount = 0;
			std::unordered_map<std::string, T> epochMetrics;

			for (size_t i = 0; i < numSamples; i += batchSize) {
				size_t end = std::min(i + batchSize, numSamples);

				TensorCore::Tensor<T> batchX = inputs.SliceRows(i, end);
				TensorCore::Tensor<T> batchY = targets.SliceRows(i, end);

				// Forward propogation
				auto pred = m_Model.Forward(batchX);

				// Loss
				auto loss = m_LossFn(pred, batchY);

				// Metrics
				auto metrics = ComputeMetrics(pred, batchY);

				for (const auto& [name, value] : metrics) {
					epochMetrics[name] += value;
				}

				// Backward
				m_Optimizer.ZeroGrad();
				loss.Backward();
				m_Optimizer.Step();

				epochLoss += loss[0];
				batchCount++;

				if (OnEpochEval) {
					OnEpochEval(epoch, pred, batchY);
				}
			}

			epochLoss /= static_cast<T>(batchCount);

			for (auto& [name, value] : epochMetrics) {
				value /= static_cast<T>(batchCount);
			}

			if (OnEpochEnd) {
				TensorCore::Tensor<T> avgLoss{ {1}, inputs.GetAllocator() };
				avgLoss[0] = epochLoss;
				OnEpochEnd(epoch, avgLoss);

				if (epoch % 500 == 0) {
					std::cout
						<< "Epoch "
						<< epoch
						<< " | Train Loss: "
						<< avgLoss[0];

					for (const auto& [name, value] : epochMetrics) {
						std::cout
							<< " | "
							<< name
							<< ": "
							<< value;
					}

					std::cout << '\n';
				}
			}
		}
	}

	template <typename T>
	void Trainer<T>::Fit(const TensorCore::Tensor<T>& trainInputs, const TensorCore::Tensor<T>& trainTargets, const TensorCore::Tensor<T>& valInputs, const TensorCore::Tensor<T>& valTargets, int epochs, size_t batchSize) {
		m_Model.Train();

		if ((trainInputs.Dims()[0] != trainTargets.Dims()[0]) || (valInputs.Dims()[0] != valTargets.Dims()[0])) {
			throw std::runtime_error(
				"Input/target sample mismatch"
			);
		}

		if (batchSize == 0) {
			throw std::runtime_error(
				"Batch size cannot be zero"
			);
		}

		size_t numSamples = trainInputs.Dims()[0];

		for (int epoch = 0; epoch < epochs; ++epoch) {
			T epochLoss = 0;
			size_t batchCount = 0;
			std::unordered_map<std::string, T> epochMetrics;

			// Training Loop
			for (size_t i = 0; i < numSamples; i += batchSize) {
				size_t end = std::min(i + batchSize, numSamples);

				TensorCore::Tensor<T> batchX = trainInputs.SliceRows(i, end);
				TensorCore::Tensor<T> batchY = trainTargets.SliceRows(i, end);

				// Forward propogation
				auto pred = m_Model.Forward(batchX);

				// Loss
				auto loss = m_LossFn(pred, batchY);

				// Metrics
				auto metrics = ComputeMetrics(pred, batchY);

				for (const auto& [name, value] : metrics) {
					epochMetrics[name] += value;
				}

				// Backward
				m_Optimizer.ZeroGrad();
				loss.Backward();
				m_Optimizer.Step();

				epochLoss += loss[0];
				batchCount++;

				if (OnEpochEval) {
					OnEpochEval(epoch, pred, batchY);
				}
			}

			epochLoss /= static_cast<T>(batchCount);

			for (auto& [name, value] : epochMetrics) {
				value /= static_cast<T>(batchCount);
			}

			auto valLoss = Evaluate(valInputs, valTargets, batchSize);

			if (OnEpochEnd) {
				TensorCore::Tensor<T> avgTrainLoss{ {1}, trainInputs.GetAllocator() };
				avgTrainLoss[0] = epochLoss;

				OnEpochEnd(epoch, avgTrainLoss);

				if (epoch % 500 == 0) {
					std::cout
						<< "Epoch "
						<< epoch
						<< " | Train Loss: "
						<< avgTrainLoss[0];

					for (const auto& [name, value] : epochMetrics) {
						std::cout
							<< " | "
							<< name
							<< ": "
							<< value;
					}

					std::cout << '\n';
				}
			}
		}
	}

	template<typename T>
	void Trainer<T>::AddMetric(const std::string& name, MetricFn<T> metric) {
		m_Metrics[name] = std::move(metric);
	}

	template<typename T>
	TensorCore::Tensor<T> Trainer<T>::Evaluate(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, size_t batchSize) {
		m_Model.Evaluate();

		T totalLoss = 0;
		size_t batches = 0;

		if (inputs.Dims()[0] != targets.Dims()[0]) {
			throw std::runtime_error(
				"Input/target sample mismatch"
			);
		}

		if (batchSize == 0) {
			throw std::runtime_error(
				"Batch size cannot be zero"
			);
		}

		size_t numSamples = inputs.Dims()[0];

		for (size_t i = 0; i < numSamples; i += batchSize) {
			size_t end = std::min(i + batchSize, numSamples);

			TensorCore::Tensor<T> batchX = inputs.SliceRows(i, end);
			TensorCore::Tensor<T> batchY = targets.SliceRows(i, end);

			// Forward propogation
			auto pred = m_Model.Forward(batchX);

			// Loss
			auto loss = m_LossFn(pred, batchY);

			totalLoss += loss[0];
			batches++;
		}

		TensorCore::Tensor<T> avgLoss{ {1}, inputs.GetAllocator() };
		avgLoss[0] = totalLoss / static_cast<T>(batches);

		m_Model.Train();

		return avgLoss;
	}

	template<typename T>
	std::unordered_map<std::string, T> Trainer<T>::ComputeMetrics(const TensorCore::Tensor<T>& pred, const TensorCore::Tensor<T>& target) {
		std::unordered_map<std::string, T> results;

		for (const auto& [name, metricFn] : m_Metrics) {
			results[name] = metricFn(pred, target);
		}

		return results;
	}
}