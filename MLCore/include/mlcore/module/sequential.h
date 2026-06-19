 /// sequential.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {
	/// <summary>
	/// A container module that holds and applies a sequence of sub-modules to an input tensor in order.
	/// </summary>
	/// <typeparam name="T">The element type used by the modules and tensors handled by this Sequential container.</typeparam>
	template <typename T>
	class Sequential : public Module<T> {
	public:
		/// <summary>
		/// Default constructor for Sequential. Creates a Sequential object using the compiler-generated default implementation.
		/// </summary>
		Sequential() = default;

		/// <summary>
		/// Constructs a new module of type ModuleType in-place using the forwarded arguments, assigns it the given name, and adds it to the Sequential container, transferring ownership.
		/// </summary>
		/// <typeparam name="ModuleType">The concrete Module<T> type to construct and add.</typeparam>
		/// <typeparam name="Args">The parameter pack of argument types forwarded to ModuleType's constructor.</typeparam>
		/// <param name="name">The name to associate with the newly constructed module inside the Sequential container.</param>
		/// <param name="args">Constructor arguments forwarded (perfectly forwarded) to create the ModuleType instance; after the call the Sequential takes ownership of the created module.</param>
		template <typename ModuleType, typename... Args>
		void EmplaceNamed(const std::string& name, Args&&... args);

		/// <summary>
		/// Constructs a new module of type ModuleType in-place using the forwarded arguments and adds it to the Sequential container, transferring ownership.
		/// </summary>
		/// <typeparam name="ModuleType">The concrete Module<T> type to construct and add.</typeparam>
		/// <typeparam name="Args">The parameter pack of argument types forwarded to ModuleType's constructor.</typeparam>
		/// <param name="args">Constructor arguments forwarded (perfectly forwarded) to create the ModuleType instance; after the call the Sequential takes ownership of the created module.</param>
		template <typename ModuleType, typename... Args>
		void Emplace(Args&&... args);

		/// <summary>
		/// Adds a module to the Sequential container, forwarding the call to the base Module<T>::Add.
		/// </summary>
		/// <typeparam name="T">Type parameter for modules stored in the container (the module's data/type parameter).</typeparam>
		/// <param name="name">The name to associate with the module.</param>
		/// <param name="mod">A unique_ptr to the Module<T> to add; ownership is transferred (moved) into the container.</param>
		void Add(const std::string& name, std::unique_ptr<Module<T>> mod);

		/// <summary>
		/// Adds a module to this Sequential container by transferring ownership of the provided unique_ptr.
		/// </summary>
		/// <typeparam name="T">The value type used by the Module and Sequential instantiations.</typeparam>
		/// <param name="mod">A std::unique_ptr to a Module<T> to add. Ownership is transferred (moved) into the Sequential; after the call the local pointer is null.</param>
		void Add(std::unique_ptr<Module<T>> mod);

		/// <summary>
		/// Applies each registered submodule's Forward method to the input tensor in order and returns the final output tensor.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensors (for example float or double).</typeparam>
		/// <param name="input">The input tensor to process (passed by const reference).</param>
		/// <returns>A new tensor containing the result after sequentially applying all submodules; the original input is not modified.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;
	};
}

#include "sequential.inl"