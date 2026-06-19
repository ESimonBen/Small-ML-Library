 /// sgd.h
#pragma once
#include "optimizer.h"
#include <unordered_map>

namespace MLCore::Optimizers {
	/// <summary>
	/// Stochastic Gradient Descent (SGD) optimizer for updating neural network parameters.
	/// </summary>
	/// <typeparam name="T">Numeric type used for parameters and computations (for example, float or double).</typeparam>
	template <typename T>
	class SGD : public Optimizer<T> {
	public:
		/// <summary>
		/// Constructs an SGD optimizer initialized with the given parameters, learning rate, and weight decay.
		/// </summary>
		/// <typeparam name="T">Numeric type used for parameter values and optimizer computations (e.g., float or double).</typeparam>
		/// <param name="params">A vector of references to NN::Parameter<T> objects that this optimizer will update.</param>
		/// <param name="learningRate">Initial learning rate used for parameter updates.</param>
		/// <param name="weightDecay">Weight decay (L2 regularization) factor applied during updates.</param>
		SGD(std::vector<std::reference_wrapper<NN::Parameter<T>>> params, T learningRate, T weightDecay = static_cast<T>(0));
		
		/// <summary>
		/// Constructs an SGD optimizer that initializes the base Optimizer with the provided parameters, learning rate, and weight decay.
		/// </summary>
		/// <typeparam name="T">The numeric type used for parameters and computations (e.g., float or double).</typeparam>
		/// <param name="params">A reference to a vector of NN::Parameter<T> representing the model parameters to be optimized.</param>
		/// <param name="learningRate">The learning rate (step size) used for parameter updates.</param>
		/// <param name="weightDecay">The weight decay (L2 regularization) factor applied during optimization.</param>
		SGD(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay = static_cast<T>(0));
		
		/// <summary>
		/// Constructs an SGD optimizer initialized with the provided parameter groups.
		/// </summary>
		/// <typeparam name="T">The numeric type of the parameters and optimizer state (for example, float or double).</typeparam>
		/// <param name="groups">A vector of ParameterGroup<T> objects that define the parameters and associated hyperparameters for optimization. The vector is passed by value and forwarded to the base Optimizer.</param>
		SGD(std::vector<ParameterGroup<T>> groups);

		/// <summary>
		/// Performs a single SGD optimization step: clips gradients, then iterates over each parameter group and updates each parameter in place using the group's learningRate and optional weightDecay.
		/// </summary>
		/// <typeparam name="T">Numeric type used for parameter and gradient values (e.g., float or double).</typeparam>
		virtual void Step() override;

		/// <summary>
		/// Returns the class/type name for this SGD instance.
		/// </summary>
		/// <typeparam name="T">The template parameter for the SGD class.</typeparam>
		/// <returns>A std::string containing the literal "SGD".</returns>
		virtual std::string TypeName() const override;

		/// <summary>
		/// Serializes the SGD optimizer state to a binary writer by writing the number of parameter groups and, for each group, its learning rate and weight decay.
		/// </summary>
		/// <typeparam name="T">Numeric type used by the model and optimizer parameters (e.g., float or double).</typeparam>
		/// <param name="writer">Binary writer to which the optimizer state is serialized.</param>
		/// <param name="model">The associated neural network module (provided for interface consistency; not used by this implementation).</param>
		virtual void SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const override;

		/// <summary>
		/// Loads the optimizer state for this SGD instance from a binary reader, restoring per-parameter-group settings. Note: Optimizer parameter groups must be reconstructed in the same order before loading state.
		/// </summary>
		/// <typeparam name="T">The data type used for model and optimizer parameters (e.g., float or double).</typeparam>
		/// <param name="reader">A BinaryReader used to read saved optimizer data (reads the number of parameter groups, then each group's learningRate and weightDecay).</param>
		/// <param name="model">The neural network module whose parameters correspond to the optimizer (provided for compatibility; not used directly in this implementation).</param>
		virtual void LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) override;
	};

	/// <summary>
	/// Stochastic Gradient Descent optimizer with momentum, dampening, and Nesterov Accelerated Gradient.
	/// </summary>
	/// <typeparam name="T">Numeric type used for parameters and computations (for example, float or double).</typeparam>
	template <typename T>
	class SGDMomentum : public Optimizer<T> {
	public:
		/// <summary>
		/// Constructs an SGDMomentum optimizer configured with the given parameters and hyperparameters. Delegates base initialization to Optimizer<T>, stores momentum-related settings, and allocates zero-initialized velocity tensors for each parameter.
		/// </summary>
		/// <typeparam name="T">Numeric type used for parameters, hyperparameters, and tensor values (e.g., float or double).</typeparam>
		/// <param name="params">A vector of reference_wrappers to NN::Parameter<T> representing the parameters to optimize. Ownership remains with the caller; the constructor registers these parameters and creates corresponding velocity tensors.</param>
		/// <param name="learningRate">The initial learning rate used by the optimizer.</param>
		/// <param name="momentum">The momentum coefficient that controls the contribution of past updates.</param>
		/// <param name="weightDecay">The weight decay (L2 regularization) coefficient applied to parameters (forwarded to the base Optimizer).</param>
		/// <param name="dampening">The dampening factor applied to momentum updates.</param>
		/// <param name="nesterov">If true, use Nesterov momentum; otherwise use standard (classical) momentum.</param>
		SGDMomentum(std::vector<std::reference_wrapper<NN::Parameter<T>>> params, T learningRate, T momentum, T weightDecay = static_cast<T>(0), T dampening = static_cast<T>(0), bool nesterov = false);
		
		/// <summary>
		/// Constructs an SGDMomentum optimizer, initializes optimizer state (learning rate, weight decay, momentum, dampening, Nesterov flag) and allocates zero-initialized velocity tensors for each parameter.
		/// </summary>
		/// <typeparam name="T">Numeric type for parameters and optimizer state (e.g., float or double).</typeparam>
		/// <param name="params">Reference to a vector of NN::Parameter<T> objects to be managed by the optimizer. These parameters are registered with the base Optimizer and used to create per-parameter velocity tensors.</param>
		/// <param name="learningRate">Initial learning rate for the optimizer.</param>
		/// <param name="momentum">Momentum coefficient used to scale the velocity update.</param>
		/// <param name="weightDecay">Weight decay (L2 regularization) coefficient applied by the base optimizer.</param>
		/// <param name="dampening">Dampening factor applied to momentum updates.</param>
		/// <param name="nesterov">If true, enables Nesterov momentum; otherwise standard momentum is used.</param>
		SGDMomentum(std::vector<NN::Parameter<T>>& params, T learningRate, T momentum, T weightDecay = static_cast<T>(0), T dampening = static_cast<T>(0), bool nesterov = false);
		
		/// <summary>
		/// Constructs an SGDMomentum optimizer using the provided parameter groups and momentum settings. Initializes internal momentum, dampening and Nesterov flags, and creates zero-initialized per-parameter velocity tensors stored in m_Velocities keyed by each parameter's ID.
		/// </summary>
		/// <typeparam name="T">Numeric type for parameters and tensors (for example float or double).</typeparam>
		/// <param name="groups">A vector of ParameterGroup<T> describing the parameter sets to optimize. Passed by value and forwarded to the base Optimizer to initialize the optimizer's parameter groups.</param>
		/// <param name="momentum">Momentum coefficient of type T used to scale the velocity term.</param>
		/// <param name="dampening">Dampening factor of type T applied to reduce the effect of momentum.</param>
		/// <param name="nesterov">Boolean flag indicating whether to use Nesterov momentum.</param>
		SGDMomentum(std::vector<ParameterGroup<T>> groups, T momentum, T dampening = static_cast<T>(0), bool nesterov = false);

		/// <summary>
		/// Performs a single optimizer step of stochastic gradient descent with momentum across all parameter groups. For each parameter it applies gradient clipping (already performed), optional weight decay, dampening, momentum updates to the stored velocity, and then updates the parameter in place using either standard or Nesterov momentum. Parameters that do not require gradients or have no gradients are skipped. Throws std::runtime_error if the optimizer velocity state for a parameter is missing.
		/// </summary>
		/// <typeparam name="T">Scalar type used for parameter values, gradients, velocities, and learning hyperparameters (for example, float or double).</typeparam>
		virtual void Step() override;

		/// <summary>
		/// Returns the type name of the SGDMomentum class.
		/// </summary>
		/// <typeparam name="T">The template parameter of the SGDMomentum class (e.g., the numeric or data type used by the class).</typeparam>
		/// <returns>A std::string containing the class type name "SGDMomentum".</returns>
		virtual std::string TypeName() const override;

		/// <summary>
		/// Serializes the optimizer's internal state — momentum, dampening, Nesterov flag, parameter groups (learning rate and weight decay), parameter names, and per-parameter velocity tensors — into the provided binary writer. Throws a runtime_error if a velocity for any parameter is missing.
		/// </summary>
		/// <typeparam name="T">Element type of tensors and parameters (e.g., float or double) used for velocities and other numeric values.</typeparam>
		/// <param name="writer">Binary writer used to emit the serialized optimizer state (writes scalars, counts, parameter name lengths and bytes, and tensor data).</param>
		/// <param name="model">The neural network model whose named parameters are used to map parameter IDs to parameter names when writing each parameter entry.</param>
		virtual void SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const override;

		/// <summary>
		/// Loads the optimizer state from a binary reader into this SGDMomentum instance. This reads momentum, dampening, the Nesterov flag, the number of parameter groups, each group's learning rate and weight decay, parameter names, and their velocity tensors, and stores them into the optimizer's member state. Throws std::runtime_error if group counts, parameter counts, a named parameter, or a velocity entry cannot be found or do not match. Note: Optimizer parameter groups must be reconstructed in the same order before loading state.
		/// </summary>
		/// <typeparam name="T">The numeric type used by the optimizer and tensors (for example float or double).</typeparam>
		/// <param name="reader">Serialization::BinaryReader used to deserialize the optimizer state (momentum, dampening, Nesterov flag, group metadata, parameter names, and velocity tensors).</param>
		/// <param name="model">NN::Module<T>& representing the model whose named parameters are used to resolve serialized parameter names to actual parameters and their IDs.</param>
		virtual void LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) override;

	private:
		T m_Momentum; /// Momentum used to contribute to updates
		T m_Dampening; /// Dampening factor applied to momentum updates
		bool m_Nesterov; /// Flag indicating whether to use Nesterov momentum
		std::unordered_map<NN::ParamID, TensorCore::Tensor<T>> m_Velocities; /// Map of velocities for each parameter
	};
}

#include "sgd.inl"