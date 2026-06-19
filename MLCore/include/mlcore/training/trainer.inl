 /// trainer.inl
#include <mlCore/data/tensorDataset.h>
#include "trainer.h"

namespace MLCore::Training {
	template <typename T>
	Trainer<T>::Trainer(NN::Module<T>& model, Optimizers::Optimizer<T>& optimizer, LossFn<T> lossFn)
		: m_Model(model), m_Optimizer(optimizer), m_LossFn(lossFn)
	{}
	
	template <typename T>
	void Trainer<T>::Fit(Data::DataLoader<T>& dataLoader, int epochs) {
		m_Model.Train();

		int endEpoch = m_CurrentEpoch + epochs;

		for (; m_CurrentEpoch < endEpoch; ++m_CurrentEpoch) {
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
				m_GlobalStep++;

				if (m_Scheduler && m_SchedulerMode == SchedulerStepMode::Batch) {
					m_Scheduler->UpdateLR();
				}

				epochLoss += loss[0];
				batchCount++;

				if (OnBatchEnd) {
					OnBatchEnd(m_CurrentEpoch, pred, y);
				}
			}

			if (batchCount <= 0) {
				throw std::runtime_error("ERROR: No batches");
			}

			epochLoss /= static_cast<T>(batchCount);

			for (auto& [name, value] : epochMetrics) {
				value /= static_cast<T>(batchCount);
			}

			if (OnEpochEnd) {
				EpochStats<T> stats;

				stats.epoch = m_CurrentEpoch;
				stats.trainLoss = epochLoss;
				stats.trainMetrics = std::move(epochMetrics);

				const std::vector<Optimizers::ParameterGroup<T>>& paramGroups = m_Optimizer.ParamGroups();

				for (const Optimizers::ParameterGroup<T>& group : paramGroups) {
					stats.learningRates.push_back(group.learningRate);
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

		int endEpoch = m_CurrentEpoch + epochs;

		for (; m_CurrentEpoch < endEpoch; ++m_CurrentEpoch) {
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
				m_GlobalStep++;

				if (m_Scheduler && m_SchedulerMode == SchedulerStepMode::Batch) {
					m_Scheduler->UpdateLR();
				}

				epochLoss += loss[0];
				batchCount++;

				if (OnBatchEnd) {
					OnBatchEnd(m_CurrentEpoch, pred, y);
				}
			}

			if (batchCount <= 0) {
				throw std::runtime_error("ERROR: No batches");
			}

			epochLoss /= static_cast<T>(batchCount);

			for (auto& [name, value] : epochMetrics) {
				value /= static_cast<T>(batchCount);
			}

			auto valResult = Evaluate(valLoader);

			if (OnEpochEnd) {
				EpochStats<T> stats;

				if (!valResult.metrics.empty()) {
					auto it = valResult.metrics.begin();

					T metric = it->second;

					if (!m_HasBestMetric || valResult.loss < m_BestValidationMetric) {
						m_BestValidationMetric = valResult.loss;
						m_HasBestMetric = true;
					}
				}

				stats.epoch = m_CurrentEpoch;
				stats.trainLoss = epochLoss;
				stats.valLoss = valResult.loss;
				stats.trainMetrics = std::move(epochMetrics);
				stats.valMetrics = std::move(valResult.metrics);

				const std::vector<Optimizers::ParameterGroup<T>>& paramGroups = m_Optimizer.ParamGroups();

				for (const Optimizers::ParameterGroup<T>&group : paramGroups) {
					stats.learningRates.push_back(group.learningRate);
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
	
	template<typename T>
	TrainerState<T> Trainer<T>::GetState() const {
		TrainerState<T> state;

		state.currentEpoch = m_CurrentEpoch;
		state.globalStep = m_GlobalStep;
		state.bestValidationMetric = m_BestValidationMetric;
		state.hasBestMetric = m_HasBestMetric;

		return state;
	}
	
	template<typename T>
	void Trainer<T>::LoadState(const TrainerState<T>& state) {
		m_CurrentEpoch = state.currentEpoch;
		m_GlobalStep = state.globalStep;
		m_BestValidationMetric = state.bestValidationMetric;
		m_HasBestMetric = state.hasBestMetric;
	}
	
	template <typename T>
	EvaluationResult<T> Trainer<T>::Evaluate(Data::DataLoader<T>& dataLoader) {
		bool wasTraining = m_Model.IsTraining();

		m_Model.Evaluate();

		dataLoader.Reset(false);

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

		if (batches <= 0) {
			throw std::runtime_error("ERROR: No batches");
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