/// adam.h
#pragma once
#include "optimizer.h"
#include <unordered_map>
#include <mlCore/module/module.h>

namespace MLCore::Optimizers {
	/// <summary>
	/// Adam optimizer implementing the adaptive moment estimation algorithm for updating neural network parameters.
	/// </summary>
	/// <typeparam name="T">Numeric type used for parameter values, gradients, and internal accumulators (e.g., float or double).</typeparam>
	template <typename T>
	class Adam : public Optimizer<T> {
	public:
		/// <summary>
		/// Constructs an Adam optimizer for neural network parameters and initializes optimizer state (first and second moment tensors, beta powers, and timestep).
		/// </summary>
		/// <typeparam name="T">Numeric type used for parameter values and optimizer state (e.g., float or double).</typeparam>
		/// <param name="params">A vector of reference_wrappers to NN::Parameter<T> objects to be optimized. The constructor registers each parameter and creates corresponding optimizer state tensors.</param>
		/// <param name="learningRate">The initial learning rate used by the optimizer.</param>
		/// <param name="weightDecay">Weight decay (L2 regularization) coefficient applied to parameters.</param>
		/// <param name="beta1">The exponential decay rate for the first moment estimates.</param>
		/// <param name="beta2">The exponential decay rate for the second moment estimates.</param>
		/// <param name="epsilon">A small constant added for numerical stability when normalizing updates.</param>
		Adam(std::vector <std::reference_wrapper<NN::Parameter<T>>> &params, T learningRate = static_cast<T>(.001), T weightDecay = static_cast<T>(0), T beta1 = static_cast<T>(.9),
			T beta2 = static_cast<T>(.999), T epsilon = static_cast<T>(1e-8));

		/// <summary>
		/// Constructs an Adam optimizer for the given parameters and initializes internal optimizer state (first and second moment estimates) and hyperparameters.
		/// </summary>
		/// <typeparam name="T">The numeric scalar type for parameters and optimizer state (typically a floating-point type such as float or double).</typeparam>
		/// <param name="params">A vector of references to NN::Parameter<T> objects to be optimized. The constructor registers these parameters with the base Optimizer and prepares per-parameter optimizer state.</param>
		/// <param name="learningRate">The initial learning rate (step size) used by the optimizer.</param>
		/// <param name="weightDecay">The weight decay (L2 regularization) factor applied during updates.</param>
		/// <param name="beta1">The exponential decay rate for the first moment estimates (momentum term).</param>
		/// <param name="beta2">The exponential decay rate for the second moment estimates (RMS term).</param>
		/// <param name="epsilon">A small constant added for numerical stability when normalizing by the second moment.</param>
		Adam(std::vector<NN::Parameter<T>>& params, T learningRate = static_cast<T>(.001), T weightDecay = static_cast<T>(0), T beta1 = static_cast<T>(.9),
			 T beta2 = static_cast<T>(.999), T epsilon = static_cast<T>(1e-8));

		/// <summary>
		/// Constructor for an Adam optimizer specialized on type T. Initializes optimizer state, hyperparameters, and per-parameter first- and second-moment tensors.
		/// </summary>
		/// <typeparam name="T">The numeric type used for parameters, moments, and internal computations (e.g., float or double).</typeparam>
		/// <param name="groups">A vector of ParameterGroup<T> describing the parameter groups to optimize. These are forwarded to the base Optimizer<T> constructor and the contained parameters are used to allocate optimizer state.</param>
		/// <param name="beta1">The exponential decay rate for the first moment estimates (typically in [0,1)).</param>
		/// <param name="beta2">The exponential decay rate for the second moment estimates (typically in [0,1)).</param>
		/// <param name="epsilon">A small constant added for numerical stability to avoid division by zero.</param>
		Adam(std::vector<ParameterGroup<T>> groups, T beta1 = static_cast<T>(.9), T beta2 = static_cast<T>(.999), T epsilon = static_cast<T>(1e-8));

		/// <summary>
		/// Performs one optimization step using the Adam algorithm: increments the timestep, clips gradients, updates first and second moment estimates, applies bias correction and weight decay, and updates model parameters in place. Throws an exception if optimizer state for a parameter is missing.
		/// </summary>
		/// <typeparam name="T">Numeric type used for tensor elements and internal computations (e.g., float or double).</typeparam>
		virtual void Step() override;

		/// <summary>
		/// Returns the type name for this Adam instance.
		/// </summary>
		/// <typeparam name="T">The template parameter of the Adam class. It specifies the type parameter for the class template and is not used by this method.</typeparam>
		/// <returns>A std::string containing the literal "Adam".</returns>
		virtual std::string TypeName() const override;

		/// <summary>
		/// Serialize the optimizer state for the Adam instance and write it to a binary writer. This includes optimizer hyperparameters, parameter group settings, parameter names, and per-parameter first and second moment tensors.
		/// </summary>
		/// <typeparam name="T">The numeric type used for model parameters and optimizer state tensors (for example float or double).</typeparam>
		/// <param name="writer">A BinaryWriter used to write serialized data to an output stream or file.</param>
		/// <param name="model">A constant reference to the neural network module whose named parameters are used to map parameter IDs to parameter names for serialization.</param>
		virtual void SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const override;
		
		/// <summary>
		/// Loads the internal state of the Adam optimizer from a binary reader, restoring optimizer hyperparameters, parameter groups, and per-parameter first and second moments. Ensures the saved state matches the provided model's parameters and current optimizer groups. Note: Optimizer parameter groups must be reconstructed in the same order before loading state.
		/// </summary>
		/// <typeparam name="T">The numeric scalar type of tensors and optimizer state (e.g., float or double) used by the optimizer and model.</typeparam>
		/// <param name="reader">A BinaryReader used to read the serialized optimizer state (hyperparameters, parameter groups, names, and tensors).</param>
		/// <param name="model">The neural network module whose named parameters are used to map saved parameter names to the current parameters in order to restore per-parameter moments.</param>
		virtual void LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) override;

	private:
		T m_Beta1; /// The exponential decay rate for the first moment estimates(moving average of gradients).
		T m_Beta2; ///  The exponential decay rate for the second moment estimates (moving average of squared gradients).
		T m_BetaPow1; /// Multiplier used with m_Beta1.
		T m_BetaPow2; /// Multiplier used with m_Beta2.
		T m_Epsilon; /// Small constant added for numerical stability when computing updates.
		size_t m_Timestep; /// Stores a timestep value.
		std::unordered_map<NN::ParamID, TensorCore::Tensor<T>> m_FirstMoment; /// Map of first moments for each parameter.
		std::unordered_map<NN::ParamID, TensorCore::Tensor<T>> m_SecondMoment; /// Map of second moments for each parameter.
	};

	/// <summary>
	/// AdamW optimizer template implementing the Adam algorithm with decoupled weight decay for training neural network parameters.
	/// </summary>
	/// <typeparam name="T">Numeric type used for optimizer calculations (e.g., float or double).</typeparam>
	template <typename T>
	class AdamW : public Optimizer<T> {
	public:
		/// <summary>
		/// Constructs and initializes an AdamW optimizer instance, setting optimizer hyperparameters and allocating per-parameter first and second moment tensors (initialized to zero).
		/// </summary>
		/// <typeparam name="T">Numeric type for parameter values and optimizer state (e.g., float or double).</typeparam>
		/// <param name="params">A non-const reference to a vector of reference_wrappers for NN::Parameter<T> objects representing the model parameters to optimize. The constructor registers these parameters with the optimizer and initializes state for each.</param>
		/// <param name="learningRate">Initial learning rate used by the optimizer.</param>
		/// <param name="weightDecay">Weight decay (L2 regularization) coefficient applied to parameters.</param>
		/// <param name="beta1">Exponential decay rate for the first moment estimates (beta1).</param>
		/// <param name="beta2">Exponential decay rate for the second moment estimates (beta2).</param>
		/// <param name="epsilon">Small constant added for numerical stability when computing updates.</param>
		AdamW(std::vector <std::reference_wrapper<NN::Parameter<T>>>& params, T learningRate = static_cast<T>(.001), T weightDecay = static_cast<T>(0), T beta1 = static_cast<T>(.9),
			T beta2 = static_cast<T>(.999), T epsilon = static_cast<T>(1e-8));

		/// <summary>
		/// Constructs an AdamW optimizer and initializes internal state (hyperparameters, timestep, and per-parameter first and second moment tensors).
		/// </summary>
		/// <typeparam name="T">The numeric type used for parameters, moment tensors, and optimizer state (e.g., float or double).</typeparam>
		/// <param name="params">A vector of references to model parameters to optimize; the constructor registers each parameter and allocates zero-initialized first- and second-moment tensors for them.</param>
		/// <param name="learningRate">The initial learning rate used by the optimizer.</param>
		/// <param name="weightDecay">The weight decay (L2 regularization) coefficient applied during optimization.</param>
		/// <param name="beta1">The exponential decay rate for the first moment estimates (moving average of gradients).</param>
		/// <param name="beta2">The exponential decay rate for the second moment estimates (moving average of squared gradients).</param>
		/// <param name="epsilon">A small constant added for numerical stability when normalizing updates.</param>
		AdamW(std::vector<NN::Parameter<T>>& params, T learningRate = static_cast<T>(.001), T weightDecay = static_cast<T>(0), T beta1 = static_cast<T>(.9),
			T beta2 = static_cast<T>(.999), T epsilon = static_cast<T>(1e-8));

		/// <summary>
		/// Constructs an AdamW optimizer instance and initializes optimizer state (first and second moment tensors) for each parameter.
		/// </summary>
		/// <typeparam name="T">The numeric type used for parameters, moments, and internal calculations (e.g., float or double).</typeparam>
		/// <param name="groups">A vector of ParameterGroup<T> describing parameter sets and their options; the constructor registers each parameter and allocates corresponding moment tensors.</param>
		/// <param name="beta1">The exponential decay rate for the first moment estimates (beta1).</param>
		/// <param name="beta2">The exponential decay rate for the second moment estimates (beta2).</param>
		/// <param name="epsilon">A small constant added for numerical stability in denominator calculations.</param>
		AdamW(std::vector<ParameterGroup<T>> groups, T beta1 = static_cast<T>(.9), T beta2 = static_cast<T>(.999), T epsilon = static_cast<T>(1e-8));

		/// <summary>
		/// Performs one optimization step using the AdamW algorithm (Adam with decoupled weight decay). The method increments the internal timestep, applies gradient clipping, updates exponential moving averages (first and second moments) with bias correction, applies weight decay if configured, initializes missing moment tensors, and updates parameter values in-place. Throws a runtime_error if optimizer state for a parameter is missing.
		/// </summary>
		/// <typeparam name="T">Numeric type used for parameters, gradients, and optimizer state (typically a floating-point type such as float or double).</typeparam>
		virtual void Step() override;

		/// <summary>
		/// Returns the name of the AdamW optimizer as a string.
		/// </summary>
		/// <typeparam name="T">The type parameter of the AdamW class template (the value type used by the optimizer).</typeparam>
		/// <returns>A std::string containing the type name "AdamW".</returns>
		virtual std::string TypeName() const override;

		/// <summary>
		/// Serializes the internal state of this AdamW optimizer to the provided binary writer for the given model. The function writes optimizer hyperparameters (beta1, beta2, beta powers, epsilon, timestep), parameter group settings (learning rate, weight decay), and for each parameter writes its name and the corresponding first and second moment tensors.
		/// </summary>
		/// <typeparam name="T">The numeric type used by the optimizer and tensors (e.g., float or double).</typeparam>
		/// <param name="writer">Binary writer used to emit the serialized bytes. The function writes numeric hyperparameters, group/parameter counts, parameter names, and tensor data via this writer.</param>
		/// <param name="model">The neural network module whose named parameters are used to map parameter IDs to human-readable names that are saved alongside each parameter's optimizer state.</param>
		virtual void SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const override;

		/// <summary>
		/// Loads the AdamW optimizer state from a binary reader and restores it into the provided model. This includes global optimizer hyperparameters (beta1, beta2, beta powers, epsilon, timestep), parameter group settings (learning rate, weight decay), and per-parameter first and second moment tensors. Throws std::runtime_error on mismatches (parameter group count or size), missing named parameters, or missing moment tensors. Note: Optimizer parameter groups must be reconstructed in the same order before loading state.
		/// </summary>
		/// <typeparam name="T">Numeric type for model parameters and optimizer tensors (for example, float or double).</typeparam>
		/// <param name="reader">Serialization::BinaryReader used to deserialize the optimizer state from a binary source.</param>
		/// <param name="model">NN::Module<T> whose named parameters are used to map saved optimizer state entries to the model's actual parameters.</param>
		virtual void LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) override;

	private:
		T m_Beta1; /// The exponential decay rate for the first moment estimates(moving average of gradients).
		T m_Beta2; ///  The exponential decay rate for the second moment estimates (moving average of squared gradients).
		T m_BetaPow1; /// Multiplier used with m_Beta1.
		T m_BetaPow2; /// Multiplier used with m_Beta2.
		T m_Epsilon; /// Small constant added for numerical stability when computing updates.
		size_t m_Timestep; /// Stores a timestep value.
		std::unordered_map<NN::ParamID, TensorCore::Tensor<T>> m_FirstMoment; /// Map of first moments for each parameter.
		std::unordered_map<NN::ParamID, TensorCore::Tensor<T>> m_SecondMoment; /// Map of second moments for each parameter.
	};
}

#include "adam.inl"