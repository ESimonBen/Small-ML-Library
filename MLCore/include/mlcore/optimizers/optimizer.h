 /// optimizer.h
#pragma once
#include <vector>
#include <functional>
#include <mlCore/tensor/tensor.h>
#include <mlCore/module/module.h>
#include <mlCore/optimizers/parameterGroup.h>
#include <mlCore/serialization/binaryArchive.h>

namespace MLCore::Optimizers {
	/// <summary>
	/// Abstract optimizer base class template that manages parameter groups and common optimization utilities (e.g., gradient clipping). Concrete optimizers implement the Step and serialization behaviors.
	/// </summary>
	/// <typeparam name="T">Numeric type used for parameters, gradients, and optimization state (typically a floating-point type).</typeparam>
	template <typename T>
	class Optimizer {
	public:
		/// <summary>
		/// Constructs an Optimizer<T> and adds a parameter group initialized with the provided parameters, learning rate, and weight decay.
		/// </summary>
		/// <typeparam name="T">The numeric type used by the parameters and optimizer (for example, float or double).</typeparam>
		/// <param name="params">A vector of std::reference_wrapper<NN::Parameter<T>> representing the parameters to manage as a single parameter group.</param>
		/// <param name="learningRate">The learning rate to use for the added parameter group.</param>
		/// <param name="weightDecay">The weight decay (L2 regularization) factor to apply to the added parameter group.</param>
		Optimizer(std::vector<std::reference_wrapper<NN::Parameter<T>>> params, T learningRate, T weightDecay = static_cast<T>(0));
		
		/// <summary>
		/// Initializes an Optimizer for the given model parameters, creating an internal parameter group that holds references to the provided parameters along with the specified learning rate and weight decay.
		/// </summary>
		/// <typeparam name="T">The numeric type of the parameters and optimizer values (for example, float or double).</typeparam>
		/// <param name="params">A vector of NN::Parameter<T> objects to optimize. The constructor stores references to these parameters (using std::reference_wrapper) rather than copying them.</param>
		/// <param name="learningRate">The initial learning rate to associate with the created parameter group.</param>
		/// <param name="weightDecay">The weight decay (L2 regularization) coefficient to associate with the created parameter group.</param>
		Optimizer(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay = static_cast<T>(0));
		
		/// <summary>
		/// Constructs an Optimizer<T>, initializing its parameter groups by moving the provided vector into the instance.
		/// </summary>
		/// <typeparam name="T">The value type used by the parameter groups and the optimizer.</typeparam>
		///	<param name="groups">A vector of ParameterGroup<T> objects to initialize the optimizer with. The contents are moved into the optimizer.</param>
		Optimizer(std::vector<ParameterGroup<T>> groups);

		/// <summary>
		/// Virtual destructor for the Optimizer class. Ensures that derived class destructors are called and resources are cleaned up correctly when deleting objects through a base-class pointer.
		/// </summary>
		virtual ~Optimizer() = default;

		/// <summary>
		/// Pure virtual function that performs a single step of processing; must be overridden by derived classes.
		/// </summary>
		virtual void Step() = 0;

		/// <summary>
		/// Resets the gradients to zero for every parameter in the optimizer's parameter groups that require gradient computation.
		/// </summary>
		/// <typeparam name="T">The element type used by parameters and tensors (for example, float or double).</typeparam>
		virtual void ZeroGrad();

		/// <summary>
		/// Returns a non-const reference to the optimizer's collection of parameter groups.
		/// </summary>
		/// <typeparam name="T">Type used for parameters stored in each ParameterGroup (e.g., a tensor or numeric type).</typeparam>
		/// <returns>A non-const reference to the internal std::vector<ParameterGroup<T>> that holds the optimizer's parameter groups. Modifying the returned vector or its elements updates the optimizer's internal state; the reference is valid while the optimizer instance exists and its container is not reallocated.</returns>
		std::vector<ParameterGroup<T>>& ParamGroups();

		/// <summary>
		/// Enables gradient clipping and sets the maximum allowed gradient norm for the optimizer.
		/// </summary>
		/// <typeparam name="T">Numeric type for the gradient norm (e.g., float or double).</typeparam>
		/// <param name="maxNorm">Maximum gradient norm to use for clipping; gradients with norm above this value will be clipped.</param>
		void SetClipGradNorm(T maxNorm);
		
		/// <summary>
		/// Const-qualified pure virtual member function that returns the optimizer's type name.
		/// </summary>
		/// <returns>A std::string containing the name of the concrete type for the optimizer.</returns>
		virtual std::string TypeName() const = 0;

		/// <summary>
		/// Pure virtual method that serializes the state of the given model into a binary writer. Implementations should write the model's persistent state to the provided writer.
		/// </summary>
		/// <param name="writer">Binary writer used to output the serialized state.</param>
		/// <param name="model">The model module whose state should be saved (provided as a const reference).</param>
		virtual void SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const = 0;

		/// <summary>
		/// Deserializes and loads state into the given neural-network module from a binary reader. This is a pure virtual method that must be implemented by derived classes.
		/// </summary>
		/// <param name="reader">Reference to a binary reader used to read the serialized state data.</param>
		/// <param name="model">Reference to the NN::Module whose state will be loaded or updated.</param>
		virtual void LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) = 0;

	protected:
		/// <summary>
		/// If gradient clipping is enabled, computes the global L2 norm of all parameter gradients and, when that norm exceeds m_MaxNorm, scales all gradients in-place to enforce the maximum norm (uses a small epsilon for numerical stability).
		/// </summary>
		/// <typeparam name="T">Numeric type used for parameter values and gradients (e.g., float or double).</typeparam>
		void ClipGradients();
		
	protected:
		std::vector<ParameterGroup<T>> m_ParamGroups; /// A container of ParameterGroups

	private:
		bool m_UseClip = false; /// Flag indicating whether clipping is enabled.
		T m_MaxNorm = static_cast<T>(0); /// The maximum L2 norm the gradient is limited to.
	};
}

#include "optimizer.inl"