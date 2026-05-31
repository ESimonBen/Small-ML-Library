// trainer.h
#pragma once
#include <functional>
#include <mlCore/tensor/tensor.h>
#include <mlCore/module/module.h>
#include <mlCore/data/dataLoader.h>
#include <mlCore/optimizers/optimizer.h>
#include <mlCore/schedulers/lrScheduler.h>

namespace MLCore::Training {
	template <typename T>
	using LossFn = std::function<TensorCore::Tensor<T>(const TensorCore::Tensor<T>&, const TensorCore::Tensor<T>&)>;

	template <typename T>
	using MetricFn = std::function<T(const TensorCore::Tensor<T>& pred, const TensorCore::Tensor<T>& target)>;

	template <typename T>
	struct EpochStats {
		int epoch = 0;
		T trainLoss = static_cast<T>(0);
		T valLoss = static_cast<T>(0);
		T learningRate = static_cast<T>(0);
		std::unordered_map<std::string, T> trainMetrics;
		std::unordered_map<std::string, T> valMetrics;
	};

	template <typename T>
	struct EvaluationResult {
		T loss = static_cast<T>(0);
		std::unordered_map<std::string, T> metrics;
	};

	enum class SchedulerStepMode {
		Epoch, Batch
	};

	template <typename T>
	class Trainer {
	public:
		Trainer(NN::Module<T>& model, Optimizers::Optimizer<T>& optimizer, LossFn<T> lossFn);

		void Fit(Data::DataLoader<T>& dataLoader, int epochs);
		void Fit(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, int epochs, size_t batchSize);

		void Fit(Data::DataLoader<T>& trainLoader, Data::DataLoader<T>& valLoader, int epochs);
		void Fit(const TensorCore::Tensor<T>& trainInputs, const TensorCore::Tensor<T>& trainTargets, const TensorCore::Tensor<T>& valInputs, const TensorCore::Tensor<T>& valTargets, int epochs, size_t batchSize);

		void SetScheduler(Schedulers::LRScheduler<T>& scheduler, SchedulerStepMode schedulerMode);
		bool HasScheduler() const;
		Schedulers::LRScheduler<T>* GetScheduler() const;

		// Adding Metrics
		void AddMetric(const std::string& name, MetricFn<T> metric);

	public:
		// Optional hooks for debugging
		std::function<void(const EpochStats<T>&)> OnEpochEnd;
		std::function<void(int epoch, const TensorCore::Tensor<T>& pred, const TensorCore::Tensor<T>& targets)> OnBatchEnd;

	private:
		EvaluationResult<T> Evaluate(Data::DataLoader<T>& dataLoader);
		EvaluationResult<T> Evaluate(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, size_t batchSize);
		std::unordered_map<std::string, T> ComputeMetrics(const TensorCore::Tensor<T>& pred, const TensorCore::Tensor<T>& target);

	private:
		NN::Module<T>& m_Model;
		Optimizers::Optimizer<T>& m_Optimizer;
		LossFn<T> m_LossFn;
		std::unordered_map<std::string, MetricFn<T>> m_Metrics;

		// Optional scheduler handle
		// Non-owning pointer (scheduler must outlive trainer)
		Schedulers::LRScheduler<T>* m_Scheduler = nullptr;
		SchedulerStepMode m_SchedulerMode = SchedulerStepMode::Epoch;
	};
}

#include "trainer.inl"