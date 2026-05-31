// trainer.inl
#include <mlCore/data/tensorDataset.h>

namespace MLCore::Training {
	template <typename T>
	Trainer<T>::Trainer(NN::Module<T>& model, Optimizers::Optimizer<T>& optimizer, LossFn<T> lossFn)
		: m_Model(model), m_Optimizer(optimizer), m_LossFn(lossFn)
	{}

	template <typename T>
	void Trainer<T>::Fit(Data::DataLoader<T>& dataLoader, int epochs) {
		m_Model.Train();

		for (int epoch = 0; epoch < epochs; ++epoch) {
			dataLoader.Reset();

			T epochLoss = static_cast<T>(0);
			size_t batchCount = 0;
			std::unordered_map<std::string, T> epochMetrics;

			while (dataLoader.HasNext()) {
				// Load data
				auto [x, y] = dataLoader.Next();

				// Forward
				auto pred = m_Model.Forward(x);

				// Loss
				auto loss = m_LossFn(pred, y);

				// Metrics
				auto metrics = ComputeMetrics(pred, y);

				for (const auto& [name, value] : metrics) {
					epochMetrics[name] += value;
				}

				// Backward
				m_Optimizer.ZeroGrad();
				loss.Backward();
				m_Optimizer.Step();

				if (m_Scheduler && m_SchedulerMode == SchedulerStepMode::Batch) {
					m_Scheduler->UpdateLR();
				}

				epochLoss += loss[0];
				batchCount++;

				if (OnBatchEnd) {
					OnBatchEnd(epoch, pred, y);
				}
			}

			epochLoss /= static_cast<T>(batchCount);

			for (auto& [name, value] : epochMetrics) {
				value /= static_cast<T>(batchCount);
			}

			if (OnEpochEnd) {
				EpochStats<T> stats;

				stats.epoch = epoch;
				stats.trainLoss = epochLoss;
				stats.trainMetrics = std::move(epochMetrics);

				if (!m_Optimizer.ParamGroups().empty()) {
					stats.learningRate = m_Optimizer.ParamGroups()[0].learningRate;
				}

				OnEpochEnd(stats);
			}

			if (m_Scheduler && m_SchedulerMode == SchedulerStepMode::Epoch) {
				m_Scheduler->UpdateLR();
			}
		}
	}

	template <typename T>
	void Trainer<T>::Fit(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, int epochs, size_t batchSize) {
		Data::TensorDataset<T> dataset{ inputs, targets };
		Data::DataLoader<T> dataLoader{ dataset, batchSize, true };

		Fit(dataLoader, epochs);
	}

	template <typename T>
	void Trainer<T>::Fit(Data::DataLoader<T>& trainLoader, Data::DataLoader<T>& valLoader, int epochs) {
		m_Model.Train();

		for (int epoch = 0; epoch < epochs; ++epoch) {
			trainLoader.Reset();

			T epochLoss = static_cast<T>(0);
			size_t batchCount = 0;
			std::unordered_map<std::string, T> epochMetrics;

			while (trainLoader.HasNext()) {
				// Load data
				auto [x, y] = trainLoader.Next();

				// Forward
				auto pred = m_Model.Forward(x);

				// Loss
				auto loss = m_LossFn(pred, y);

				// Metrics
				auto metrics = ComputeMetrics(pred, y);

				for (const auto& [name, value] : metrics) {
					epochMetrics[name] += value;
				}

				// Backward
				m_Optimizer.ZeroGrad();
				loss.Backward();
				m_Optimizer.Step();

				if (m_Scheduler && m_SchedulerMode == SchedulerStepMode::Batch) {
					m_Scheduler->UpdateLR();
				}

				epochLoss += loss[0];
				batchCount++;

				if (OnBatchEnd) {
					OnBatchEnd(epoch, pred, y);
				}
			}

			epochLoss /= static_cast<T>(batchCount);

			for (auto& [name, value] : epochMetrics) {
				value /= static_cast<T>(batchCount);
			}

			auto valResult = Evaluate(valLoader);

			if (OnEpochEnd) {
				EpochStats<T> stats;

				stats.epoch = epoch;
				stats.trainLoss = epochLoss;
				stats.valLoss = valResult.loss;
				stats.trainMetrics = std::move(epochMetrics);
				stats.valMetrics = std::move(valResult.metrics);

				if (!m_Optimizer.ParamGroups().empty()) {
					stats.learningRate = m_Optimizer.ParamGroups()[0].learningRate;
				}

				OnEpochEnd(stats);
			}

			if (m_Scheduler && m_SchedulerMode == SchedulerStepMode::Epoch) {
				m_Scheduler->UpdateLR();
			}
		}

	}

	template <typename T>
	void Trainer<T>::Fit(const TensorCore::Tensor<T>& trainInputs, const TensorCore::Tensor<T>& trainTargets, const TensorCore::Tensor<T>& valInputs, const TensorCore::Tensor<T>& valTargets, int epochs, size_t batchSize) {
		Data::TensorDataset<T> trainSet{ trainInputs, trainTargets };
		Data::TensorDataset<T> valSet{ valInputs, valTargets };

		Data::DataLoader<T> trainLoader{ trainSet, batchSize, true };
		Data::DataLoader<T> valLoader{ valSet, batchSize, false };

		Fit(trainLoader, valLoader, epochs);
	}

	template <typename T>
	void Trainer<T>::SetScheduler(Schedulers::LRScheduler<T>& scheduler, SchedulerStepMode schedulerMode) {
		m_Scheduler = &scheduler;
		m_SchedulerMode = schedulerMode;
	}

	template <typename T>
	bool Trainer<T>::HasScheduler() const {
		return m_Scheduler != nullptr;
	}

	template <typename T>
	Schedulers::LRScheduler<T>* Trainer<T>::GetScheduler() const {
		return m_Scheduler;
	}

	template<typename T>
	void Trainer<T>::AddMetric(const std::string& name, MetricFn<T> metric) {
		m_Metrics[name] = std::move(metric);
	}

	template <typename T>
	EvaluationResult<T> Trainer<T>::Evaluate(Data::DataLoader<T>& dataLoader) {
		bool wasTraining = m_Model.IsTraining();

		m_Model.Evaluate();

		dataLoader.Reset();

		EvaluationResult<T> evalResult;
		size_t batches = 0;

		while (dataLoader.HasNext()) {
			// Load data
			auto [x, y] = dataLoader.Next();

			// Forward
			auto pred = m_Model.Forward(x);

			// Loss
			auto loss = m_LossFn(pred, y);
			evalResult.loss += loss[0];

			// Metrics
			auto metrics = ComputeMetrics(pred, y);

			for (const auto& [name, value] : metrics) {
				evalResult.metrics[name] += value;
			}

			batches++;
		}

		evalResult.loss /= static_cast<T>(batches);

		for (auto& [name, value] : evalResult.metrics) {
			value /= static_cast<T>(batches);
		}

		if (wasTraining) {
			m_Model.Train();
		}
		else {
			m_Model.Evaluate();
		}

		return evalResult;
	}

	template<typename T>
	EvaluationResult<T> Trainer<T>::Evaluate(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, size_t batchSize) {
		Data::TensorDataset<T> dataset{ inputs, targets };
		Data::DataLoader<T> dataLoader{ dataset, batchSize, true };

		return Evaluate(dataLoader);
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