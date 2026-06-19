 /// module.h
#pragma once
#include <mlCore/parameters/parameter.h>

namespace MLCore::NN {
	/// <summary>
	/// Alias for a named parameter, pairing a parameter name with a reference to an NN::Parameter<T> instance.
	/// </summary>
	/// <typeparam name="T">The value type stored in the referenced NN::Parameter.</typeparam>
	template <typename T>
	using NamedParameter = std::pair<std::string, std::reference_wrapper<NN::Parameter<T>>>;

	/// <summary>
	/// Type alias for a named constant parameter: a pair of a string name and a reference to a const NN::Parameter<T>.
	/// </summary>
	/// <typeparam name="T">The type of the value stored in the NN::Parameter referenced by the alias.</typeparam>
	template <typename T>
	using ConstNamedParameter = std::pair<std::string, std::reference_wrapper<const NN::Parameter<T>>>;

	/// Class Forwarding for RegisteredModule
	template <typename T>
	class Module;

	/// <summary>
	/// Represents a registered module by storing its name and an owning pointer to a Module<T> instance.
	/// </summary>
	/// <typeparam name="T">The type parameter used by Module<T>, representing the module's contained type, configuration, or payload type.</typeparam>
	template <typename T>
	struct RegisteredModule {
		std::string name;
		std::unique_ptr<Module<T>> module;
	};

	/// <summary>
	/// Abstract base class representing a neural network module that can contain submodules, manage parameters, and perform forward evaluation in either training or evaluation mode.
	/// </summary>
	/// <typeparam name="T">Numeric type used for tensor values and parameters (e.g., float, double).</typeparam>
	template <typename T>
	class Module {
	public:
		/// <summary>
		/// Virtual defaulted destructor for Module. Ensures proper cleanup of derived classes using the compiler-generated implementation.
		/// </summary>
		virtual ~Module() = default;

		/// <summary>
		/// Performs the forward operation on the given tensor. This is a pure virtual method that derived classes must implement.
		/// </summary>
		/// <param name="input">The input tensor (const reference) to be processed.</param>
		/// <returns>A Tensor<T> containing the result of the forward computation.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const = 0;

		/// <summary>
		/// Adds a submodule to this Module, registering it under the given name and storing it in the internal submodule list.
		/// </summary>
		/// <typeparam name="T">The type parameter for Module and its submodules.</typeparam>
		/// <param name="name">The name to register the submodule under.</param>
		/// <param name="mod">A unique_ptr to the submodule. Ownership is transferred into the Module (the pointer is moved).</param>
		void Add(const std::string& name, std::unique_ptr<Module<T>> mod);

		/// <summary>
		/// Adds a Module<T> using an automatically generated layer name and takes ownership of the provided module.
		/// </summary>
		/// <typeparam name="T">The template parameter for the Module specialization.</typeparam>
		/// <param name="mod">A std::unique_ptr to the Module<T> to add; ownership is transferred into this object. The function generates a name in the form "layerN" using an internal counter (m_NameCounter) and calls the Add overload with that name and std::move(mod).</param>
		void Add(std::unique_ptr<Module<T>> mod);

		/// <summary>
		/// Collects and returns references to all parameters in this module and its submodules.
		/// </summary>
		/// <typeparam name="T">The value type of the parameters (the template parameter used by NN::Parameter).</typeparam>
		/// <returns>A std::vector of std::reference_wrapper<NN::Parameter<T>> containing non-owning references to each parameter found. The returned references refer to parameters owned by the module or its submodules and must remain valid for the lifetime of the vector's use.</returns>
		virtual std::vector<std::reference_wrapper<NN::Parameter<T>>> GetParameters();

		/// <summary>
		/// Returns a list of references to this module's parameters (including submodule parameters) without taking ownership.
		/// </summary>
		/// <typeparam name="T">The numeric or tensor element type used by the module and its parameters.</typeparam>
		/// <returns>A std::vector of std::reference_wrapper to const NN::Parameter<T>. Each entry is a non-owning reference to a parameter stored by this module or its submodules. The returned references remain valid only while the referenced modules and parameters stay alive.</returns>
		virtual std::vector<std::reference_wrapper<const NN::Parameter<T>>> GetParameters() const;

		/// <summary>
		/// Collects and returns all named parameters for this Module and its submodules.
		/// </summary>
		/// <typeparam name="T">The element type used by the Module and by NamedParameter; the type of the parameter values.</typeparam>
		/// <returns>A std::vector of NamedParameter<T> containing the named parameters collected from this module and its submodules.</returns>
		virtual std::vector<NamedParameter<T>> GetNamedParameters();

		/// <summary>
		/// Collects and returns all named parameters from this module and its submodules.
		/// </summary>
		/// <typeparam name="T">The value type of the named parameters returned (the type stored in ConstNamedParameter).</typeparam>
		/// <returns>A std::vector of ConstNamedParameter<T> containing all named parameters for this module and its submodules (read-only wrappers for parameter values).</returns>
		virtual std::vector<ConstNamedParameter<T>> GetNamedParameters()const ;

		/// <summary>
		/// Invokes the module's forward operation for the given input tensor.
		/// </summary>
		/// <typeparam name="T">The element/data type stored in the TensorCore::Tensor (e.g., float, double).</typeparam>
		/// <param name="input">A const reference to the input tensor to be processed by the module.</param>
		/// <returns>A TensorCore::Tensor<T> containing the module's output; this value is produced by calling the module's Forward method.</returns>
		TensorCore::Tensor<T> operator()(const TensorCore::Tensor<T>& input);

		/// <summary>
		/// Puts the module into training mode and invokes Train() on each registered submodule.
		/// </summary>
		/// <typeparam name="T">The type parameter for the Module template; represents the data or element type used by the module and its submodules.</typeparam>
		virtual void Train();

		/// <summary>
		/// Set the module to evaluation (inference) mode and propagate that state to all registered submodules.
		/// </summary>
		/// <typeparam name="T">Type parameter of the Module class (e.g., the element or tensor data type used by the module).</typeparam>
		virtual void Evaluate();

		/// <summary>
		/// Returns whether the module is currently in training mode.
		/// </summary>
		/// <typeparam name="T">The template parameter for the Module class (represents the module's element or data type).</typeparam>
		/// <returns>true if the module is in training mode; otherwise false.</returns>
		bool IsTraining() const;

	protected:
		/// <summary>
		/// Populates the given vector with references to this module's NN::Parameter<T> objects.
		/// </summary>
		/// <typeparam name="T">The value type used by the module's parameters (the template parameter of NN::Parameter).</typeparam>
		/// <param name="out">A reference to a vector that will be filled with std::reference_wrapper<NN::Parameter<T>>. The function modifies the vector to include references to the module's parameters.</param>
		virtual void CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out);

		/// <summary>
		/// Collects references to this module's parameters into the provided output vector.
		/// </summary>
		/// <typeparam name="T">The value type used by the Module and the NN::Parameter<T> instances.</typeparam>
		/// <param name="out">Reference to a vector that will be populated with std::reference_wrapper<const NN::Parameter<T>> referring to this module's parameters.</param>
		virtual void CollectParameters(std::vector<std::reference_wrapper<const NN::Parameter<T>>>& out) const;

		/// <summary>
		/// Collects parameters from this module and all nested submodules, appending references to the provided output vector.
		/// </summary>
		/// <typeparam name="T">The value type used by the Module and NN::Parameter; determines the parameter type stored and referenced.</typeparam>
		/// <param name="out">Reference to a vector that will be appended with references to NN::Parameter<T> objects belonging to this module and its submodules. This output parameter is modified by the function.</param>
		void CollectSubmoduleParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out);

		/// <summary>
		/// Collects this module's parameters and the parameters of all nested submodules, appending them to the provided output vector.
		/// </summary>
		/// <typeparam name="T">Type used by NN::Parameter<T>, representing the parameter value type handled by the module.</typeparam>
		/// <param name="out">Output vector that will receive references to this module's parameters and those of all nested submodules. Existing contents are preserved; parameters are appended.</param>
		void CollectSubmoduleParameters(std::vector<std::reference_wrapper<const NN::Parameter<T>>>& out) const;

		/// <summary>
		/// Collects named parameters that match the given name and appends them to the provided output vector.
		/// </summary>
		/// <typeparam name="T">The value type used by Module and NamedParameter; indicates the type stored in the named parameters.</typeparam>
		/// <param name="name">The name used to match named parameters to collect.</param>
		/// <param name="out">Output vector that will be filled with matching NamedParameter<T> objects; this parameter is modified by the function.</param>
		virtual void CollectNamedParameters(const std::string& name, std::vector<NamedParameter<T>>& out);

		/// <summary>
		/// Collects named parameters that match the given name from the module and appends them to the provided output vector. The method is const and does not modify the module.
		/// </summary>
		/// <typeparam name="T">The type parameter for the Module and for ConstNamedParameter<T>.</typeparam>
		/// <param name="name">The name used to match named parameters to collect.</param>
		/// <param name="out">Output vector that will be appended with matching ConstNamedParameter<T> entries.</param>
		virtual void CollectNamedParameters(const std::string& name, std::vector<ConstNamedParameter<T>>& out) const;

		/// <summary>
		/// Recursively collects named parameters from this module and its registered submodules, using the provided name as a prefix for child parameters.
		/// </summary>
		/// <typeparam name="T">Type of the values stored in the NamedParameter<T> objects.</typeparam>
		/// <param name="name">Prefix to apply to collected parameter names. If empty, submodule names are used as the starting prefix.</param>
		/// <param name="out">Vector that will be appended with NamedParameter<T> entries found in this module and its submodules.</param>
		void CollectNamedSubmoduleParameters(const std::string& name, std::vector<NamedParameter<T>>& out);

		/// <summary>
		/// Recursively collects named parameters from this module and all nested submodules, appending them to the output vector. Submodule parameter names are qualified by the provided name using dot notation.
		/// </summary>
		/// <typeparam name="T">The parameter value type used by this module and by ConstNamedParameter<T>.</typeparam>
		/// <param name="name">Prefix to apply to collected parameter names. If empty, submodule names are used as the top-level name; otherwise child names are appended as "parent.child".</param>
		/// <param name="out">Output vector to which found ConstNamedParameter<T> entries are appended. Existing contents are preserved.</param>
		void CollectNamedSubmoduleParameters(const std::string& name, std::vector<ConstNamedParameter<T>>& out) const;

	protected:
		std::vector<RegisteredModule<T>> m_Submodules; /// Container of named submodules
		size_t m_NameCounter = 0; /// Count of how many auto-generated names the model created
		bool m_IsTraining = true; /// Flag to indicate if the model is being trained
	};
}

#include "module.inl"