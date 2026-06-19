 /// trainer.h
#pragma once
#include <functional>
#include <mlCore/tensor/tensor.h>
#include <mlCore/module/module.h>
#include <mlCore/data/dataLoader.h>
#include <mlCore/optimizers/optimizer.h>
#include <mlCore/schedulers/lrScheduler.h>

namespace MLCore::Training {
	/// <summary>
	/// Type alias for a callable loss function that takes two tensors and produces a tensor result.
	/// </summary>
	/// <typeparam name="T">Element (scalar) type of the tensors (for example, float or double).</typeparam>
	template <typename T>
	using LossFn = std::function<TensorCore::Tensor<T>(const TensorCore::Tensor<T>&, const TensorCore::Tensor<T>&)>;

	/// <summary>
	/// Type alias for a std::function representing a metric that takes predicted and target tensors and returns a metric value of type T.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensors and the return type of the metric function.</typeparam>
	template <typename T>
	using MetricFn = std::function<T(const TensorCore::Tensor<T>& pred, const TensorCore::Tensor<T>& target)>;

	/// <summary>
	/// Holds statistics recorded for a single training epoch.
	/// </summary>
	/// <typeparam name="T">Numeric type used for losses, learning rates, and metric values (typically float or double).</typeparam>
	template <typename T>
	struct EpochStats {
		int epoch = 0; /// Integer variable that represents an epoch, initialized to 0.
		T trainLoss = static_cast<T>(0); /// Loss during training of type T
		T valLoss = static_cast<T>(0); /// Loss during validation of type T
		std::vector<T> learningRates; /// Container of learning rates for each group of parameters
		std::unordered_map<std::string, T> trainMetrics; /// Map of metrics during training
		std::unordered_map<std::string, T> valMetrics; /// Map of metrics during validation
	};

	/// <summary>
	/// Container for evaluation results that stores an overall loss and a set of named metric values.
	/// </summary>
	/// <typeparam name="T">Numeric type used for the loss and metric values (e.g., float or double).</typeparam>
	template <typename T>
	struct EvaluationResult {
		T loss = static_cast<T>(0); /// Validation loss of type T
		std::unordered_map<std::string, T> metrics; /// Evaluation metrics
	};

	/// <summary>
	/// Specifies when a scheduler should perform a step.
	/// </summary>
	enum class SchedulerStepMode {
		Epoch, Batch
	};

	/// <summary>
	/// Stores training progress and the best observed validation metric for a trainer.
	/// </summary>
	/// <typeparam name="T">Type used for the validation metric (e.g., float or double).</typeparam>
	template <typename T>
	struct TrainerState {
		int currentEpoch = 0; /// The current epoch the trainer was on
		size_t globalStep = 0; /// Global counter steps to train each batch
		T bestValidationMetric = static_cast<T>(0); /// Best metric during validation
		bool hasBestMetric = false; /// Flag determining if the trainer has a best metric
	};

	/// <summary>
	/// A reusable training helper that runs training loops for a neural network model, manages the optimizer and loss function, tracks metrics and state, and optionally integrates a learning-rate scheduler and user callbacks.
	/// </summary>
	/// <typeparam name="T">Numeric type used for model parameters and tensors (for example float or double).</typeparam>
	template <typename T>
	class Trainer {
	public:
		/// <summary>
		/// Initializes a Trainer<T> instance with a model, an optimizer, and a loss function.
		/// </summary>
		/// <typeparam name="T">Numeric/data type used by the model, optimizer, and loss computations (e.g., float or double).</typeparam>
		/// <param name="model">Reference to the neural network module to be trained; stored as the trainer's model.</param>
		/// <param name="optimizer">Reference to the optimizer used to update the model's parameters during training; stored by the trainer.</param>
		/// <param name="lossFn">Loss function (callable or functor) used to compute training loss; stored by the trainer.</param>
		Trainer(NN::Module<T>& model, Optimizers::Optimizer<T>& optimizer, LossFn<T> lossFn);

		/// <summary>
		/// Trains the model for a specified number of epochs using batches from the given DataLoader. For each batch it runs the forward pass, computes loss and metrics, performs backward and optimizer steps, updates learning rate scheduler (if configured), accumulates per-epoch loss and metrics, and calls batch/epoch callbacks. Throws std::runtime_error if an epoch contains no batches.
		/// </summary>
		/// <typeparam name="T">Numeric type used for model values, losses, and metrics (for example float or double).</typeparam>
		/// <param name="dataLoader">Reference to a Data::DataLoader<T> that supplies training batches. The loader is reset at the start of each epoch and iterated until exhausted.</param>
		/// <param name="epochs">Number of epochs to run; training proceeds from the current epoch (m_CurrentEpoch) up to m_CurrentEpoch + epochs. Each epoch processes all batches provided by dataLoader.</param>
		void Fit(Data::DataLoader<T>& dataLoader, int epochs);

		/// <summary>
		/// Creates a TensorDataset and DataLoader from the given input and target tensors and trains the Trainer for the specified number of epochs using the given batch size.
		/// </summary>
		/// <typeparam name="T">Element type stored in the tensors (for example float or double).</typeparam>
		/// <param name="inputs">Const reference to a tensor containing the input features for training.</param>
		/// <param name="targets">Const reference to a tensor containing the target values or labels corresponding to the inputs.</param>
		/// <param name="epochs">Number of epochs to run the training loop.</param>
		/// <param name="batchSize">Number of samples per batch used by the DataLoader.</param>
		void Fit(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, int epochs, size_t batchSize);

		/// <summary>
		/// Trains the model for a given number of epochs using the provided training and validation data loaders. Updates model parameters via the optimizer, optionally updates a learning-rate scheduler, computes and aggregates loss and metrics, tracks best validation metric, and invokes batch/epoch callbacks.
		/// </summary>
		/// <typeparam name="T">Numeric type used for model data, losses, and metrics (e.g., float or double).</typeparam>
		/// <param name="trainLoader">Reference to the training data loader that provides batches. The loader is reset at each epoch and must support Reset(), HasNext(), and Next().</param>
		/// <param name="valLoader">Reference to the validation data loader used to evaluate the model after each epoch.</param>
		/// <param name="epochs">Number of epochs to run starting from the current epoch (m_CurrentEpoch). If zero or negative, no epochs will be executed. If an epoch contains no batches, the function throws std::runtime_error.</param>
		void Fit(Data::DataLoader<T>& trainLoader, Data::DataLoader<T>& valLoader, int epochs);
		
		/// <summary>
		/// Prepares training and validation datasets and data loaders, then delegates training to the loader-based Fit overload.
		/// </summary>
		/// <typeparam name="T">Element type stored in the tensors (e.g., float, double).</typeparam>
		/// <param name="trainInputs">Tensor of input features for the training set.</param>
		/// <param name="trainTargets">Tensor of target values for the training set.</param>
		/// <param name="valInputs">Tensor of input features for the validation set.</param>
		/// <param name="valTargets">Tensor of target values for the validation set.</param>
		/// <param name="epochs">Number of training epochs to run.</param>
		/// <param name="batchSize">Number of samples per batch used to construct data loaders.</param>
		void Fit(const TensorCore::Tensor<T>& trainInputs, const TensorCore::Tensor<T>& trainTargets, const TensorCore::Tensor<T>& valInputs, const TensorCore::Tensor<T>& valTargets, int epochs, size_t batchSize);

		/// <summary>
		/// Sets the trainer's learning-rate scheduler and the scheduler step mode.
		/// </summary>
		/// <typeparam name="T">The numeric type used by the Trainer and the learning-rate scheduler (e.g., float or double).</typeparam>
		/// <param name="scheduler">Reference to a Schedulers::LRScheduler<T> instance to use. The trainer stores a pointer to this scheduler, so the caller must ensure the scheduler remains valid for the trainer's lifetime.</param>
		/// <param name="schedulerMode">SchedulerStepMode value that specifies when or how the scheduler should be advanced.</param>
		void SetScheduler(Schedulers::LRScheduler<T>& scheduler, SchedulerStepMode schedulerMode);

		/// <summary>
		/// Determines whether the Trainer instance has an associated scheduler.
		/// </summary>
		/// <typeparam name="T">The type parameter for the Trainer class.</typeparam>
		/// <returns>true if m_Scheduler is not null (a scheduler is present); otherwise false.</returns>
		bool HasScheduler() const;

		/// <summary>
		/// Returns the trainer's learning-rate scheduler.
		/// </summary>
		/// <typeparam name="T">Type parameter used by Trainer and the scheduler.</typeparam>
		/// <returns>Pointer to the Trainer's Schedulers::LRScheduler<T> instance. May be nullptr if no scheduler is set.</returns>
		Schedulers::LRScheduler<T>* GetScheduler() const;

		/// <summary>
		/// Adds or updates a named metric in the trainer's metrics collection.
		/// </summary>
		/// <typeparam name="T">The sample or data type used by the metric functions.</typeparam>
		/// <param name="name">The key/name under which the metric will be stored.</param>
		/// <param name="metric">A callable metric function for type T; the function object is moved into the trainer's metric storage. If a metric with the same name exists, it will be replaced.</param>
		void AddMetric(const std::string& name, MetricFn<T> metric);

		/// <summary>
		/// Returns a snapshot of the trainer's current state. The method constructs and returns a TrainerState<T> populated from the trainer's internal members (currentEpoch, globalStep, bestValidationMetric, and hasBestMetric) and does not modify the Trainer (const).
		/// </summary>
		/// <typeparam name="T">Type parameter used by Trainer and TrainerState; represents the type used for metrics and any type-dependent state.</typeparam>
		/// <returns>A TrainerState<T> object containing the trainer's currentEpoch, globalStep, bestValidationMetric, and hasBestMetric values.</returns>
		TrainerState<T> GetState() const;

		/// <summary>
		/// Restores the trainer's internal state from the provided TrainerState object.
		/// </summary>
		/// <typeparam name="T">The element or numeric type used by the Trainer and TrainerState.</typeparam>
		/// <param name="state">The source TrainerState<T> whose values (currentEpoch, globalStep, bestValidationMetric, hasBestMetric) are copied into this Trainer.</param>
		void LoadState(const TrainerState<T>& state);

	public:
		std::function<void(const EpochStats<T>&)> OnEpochEnd; /// A callback invoked when an epoch ends. Holds a callable that accepts the completed epoch's statistics.
		std::function<void(int epoch, const TensorCore::Tensor<T>& pred, const TensorCore::Tensor<T>& targets)> OnBatchEnd; /// Callback invoked at the end of a batch, receiving the current epoch and the predicted and target tensors for that batch.

	private:
		/// <summary>
		/// Runs the model in evaluation mode over all batches provided by the given data loader, computing and returning the average loss and averaged metrics. The function temporarily switches the model to evaluation mode, resets and consumes the data loader, restores the model's previous training state, and throws std::runtime_error if no batches are available.
		/// </summary>
		/// <typeparam name="T">Numeric type used for loss and metric accumulation and returned in EvaluationResult<T> (e.g., float or double).</typeparam>
		/// <param name="dataLoader">Reference to a Data::DataLoader<T> that supplies evaluation batches. The loader is reset and then consumed (HasNext/Next) during evaluation; its contents will be advanced.</param>
		/// <returns>An EvaluationResult<T> containing the mean loss and averaged metric values across all processed batches.</returns>
		EvaluationResult<T> Evaluate(Data::DataLoader<T>& dataLoader);

		/// <summary>
		/// Evaluates the trainer on the provided input and target tensors by creating a dataset and data loader with the given batch size, then delegating to the data-loader-based Evaluate overload.
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (e.g., float, double) used by the trainer and evaluation.</typeparam>
		/// <param name="inputs">Const reference to a tensor of input examples to evaluate.</param>
		/// <param name="targets">Const reference to a tensor of expected outputs/labels corresponding to the inputs.</param>
		/// <param name="batchSize">Number of samples per batch used when constructing the Data::DataLoader.</param>
		/// <returns>An EvaluationResult<T> containing the aggregated evaluation results (e.g., metrics, loss) computed over the dataset.</returns>
		EvaluationResult<T> Evaluate(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, size_t batchSize);
		
		/// <summary>
		/// Computes all configured metrics for the given predictions and targets and returns their values keyed by metric name.
		/// </summary>
		/// <typeparam name="T">Element and result type used by the tensors and metric functions (e.g., float or double).</typeparam>
		/// <param name="pred">Const reference to the tensor of predicted values.</param>
		/// <param name="target">Const reference to the tensor of target (ground-truth) values.</param>
		/// <returns>An unordered_map<string, T> that maps each metric name to its computed value of type T; contains one entry for each configured metric in m_Metrics.</returns>
		std::unordered_map<std::string, T> ComputeMetrics(const TensorCore::Tensor<T>& pred, const TensorCore::Tensor<T>& target);

	private:
		NN::Module<T>& m_Model; /// A reference to a neural network model.
		Optimizers::Optimizer<T>& m_Optimizer; /// A reference to a neural network optimizer.
		LossFn<T> m_LossFn; /// Loss function for backpropogation.
		std::unordered_map<std::string, MetricFn<T>> m_Metrics; /// Map of metrics (e.g. accuracy, lowest loss, etc.)

		Schedulers::LRScheduler<T>* m_Scheduler = nullptr; /// Non-owning pointer to scheduler (must outlive trainer)
		SchedulerStepMode m_SchedulerMode = SchedulerStepMode::Epoch; /// Holds the scheduler step mode (set to schedule LR every epoch by default).

		/// Trainer state
		int m_CurrentEpoch = 0; /// The current epoch of the trainer
		size_t m_GlobalStep = 0; /// Global counter steps to train each batch
		T m_BestValidationMetric = static_cast<T>(0); /// Best metric during validation
		bool m_HasBestMetric = false; /// Flag determining if the trainer has a best metric
	};
}

#include "trainer.inl"