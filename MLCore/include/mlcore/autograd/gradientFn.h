 /// gradientFn.h
#pragma once
#include <memory>
#include <vector>

namespace MLCore::TensorCore {
	template <typename T>
	class Tensor;
}

namespace MLCore::AutoGrad {
	/// <summary>
	/// Abstract base class that provides an interface and common utilities for gradient (backward) functions used with TensorCore::Tensor. Implementations perform backward propagation of gradients into input tensors.
	/// </summary>
	/// <typeparam name="T">Value type stored in the tensors (element type) used by TensorCore::Tensor and the gradient operations.</typeparam>
	template <typename T>
	class GradFn {
	public:
		using Impl = TensorCore::Tensor<T>::Impl;

		/// <summary>
		/// Defaulted default constructor for GradFn. Initializes the object using the compiler-provided default behavior (members are default-initialized or value-initialized as appropriate).
		/// </summary>
		GradFn() = default;

		/// <summary>
		/// Initializes a GradFn<T> by moving the provided implementation pointer into its inputs member.
		/// </summary>
		/// <param name="impl">A std::shared_ptr to the implementation object. The pointer is moved into the instance's inputs member, transferring ownership.</param>
		explicit GradFn(std::shared_ptr<Impl> impl);

		/// <summary>
		/// Constructs a GradFn<T> by taking a vector of gradient input pointers and moving it into the instance.
		/// </summary>
		/// <param name="gradInput">A vector of std::shared_ptr<Impl> representing gradient inputs. The vector is moved into the object's inputs member, transferring the container and its shared_ptr ownership.</param>
		explicit GradFn(std::vector<std::shared_ptr<Impl>> gradInput);

		/// <summary>
		/// Pure virtual method that performs the backward pass: computes gradients w.r.t. this layer/state using the provided output gradient and may allocate temporaries from the given allocator. Must be overridden by derived classes.
		/// </summary>
		/// <param name="gradOutput">Constant reference to the tensor containing the gradient of the loss with respect to this layer's output.</param>
		/// <param name="allocator">Arena-style allocator to be used for any temporary or persistent memory allocations during the backward computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) = 0;

		/// <summary>
		/// Virtual default destructor for GradFn that ensures proper cleanup of derived classes.
		/// </summary>
		virtual ~GradFn() = default;

	protected:
		/// <summary>
		/// Returns a shared pointer to the Impl object at the specified input index.
		/// </summary>
		/// <param name="i">The zero-based index of the input to retrieve.</param>
		/// <returns>A std::shared_ptr<Impl> referencing the input at the given index.</returns>
		std::shared_ptr<Impl> Input(size_t i) {
			return inputs[i];
		}

		/// <summary>
		/// Returns a constant reference to the internal vector of shared pointers to Impl.
		/// </summary>
		/// <returns>A const reference to the internal std::vector<std::shared_ptr<Impl>> that holds the inputs. The caller cannot modify the container (e.g., add or remove elements) through this reference, though the pointed-to Impl objects may be modified via the shared_ptrs if their interfaces allow it.</returns>
		const std::vector<std::shared_ptr<Impl>>& Inputs() const {
			return inputs;
		}

	protected:
		std::vector<std::shared_ptr<Impl>> inputs; /// A container of shared pointers to Impl objects representing input elements.
	};
}

#include "gradientFn.inl"