 /// tensor.h
#pragma once
#include <memory>
#include <vector>
#include <mlCore/utils/shape.h>
#include <mlCore/memory/storage.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::TensorCore {
	/// <summary>
	/// Implementation detail for a tensor object that holds shape, element storage, allocator, offset, and autograd metadata.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensor (for example, float, double, or int).</typeparam>
	template <typename T>
	struct TensorImpl {
		Utils::Shape shape; /// Declares a variable named shape of type Utils::Shape.
		Memory::Storage<T> storage; /// Declares a variable named storage of type Memory::Storage<T>.
		Memory::ArenaAllocator* allocator; /// Pointer to an ArenaAllocator instance in the Memory namespace.
		size_t offset; /// Unsigned integer value representing an offset (for example, a byte or element displacement).
		bool requiresGrad; /// Flag indicating whether gradient computation is required.
		std::shared_ptr<TensorImpl<T>> grad; /// Shared pointer to the tensor implementation that holds the gradient for a tensor of type T.
		std::shared_ptr<AutoGrad::GradFn<T>> gradFn; /// Shared pointer to an AutoGrad gradient function object for values of type T.

		/// <summary>
		/// Constructs a TensorImpl<T> with the given shape, storage, allocator and optional gradient information.
		/// </summary>
		/// <param name="shape">The tensor shape used to initialize the object's dimensions (const reference, read-only).</param>
		/// <param name="storage">The memory storage for the tensor. The storage is moved into the object.</param>
		/// <param name="allocator">Pointer to an ArenaAllocator used for memory allocations; treated as a non-owning pointer.</param>
		/// <param name="offset">Optional byte or element offset into the storage where the tensor data begins (defaults to 0).</param>
		/// <param name="requiresGrad">Whether this tensor should track gradients for automatic differentiation (defaults to false).</param>
		/// <param name="grad">Optional shared pointer to a gradient tensor associated with this tensor; moved into the object (defaults to nullptr).</param>
		/// <param name="gradFn">Optional shared pointer to the gradient function (backward function) used for autograd; moved into the object (defaults to nullptr).</param>
		TensorImpl(const Utils::Shape& shape,
			Memory::Storage<T> storage,
			Memory::ArenaAllocator* allocator,
			size_t offset = 0,
			bool requiresGrad = false,
			std::shared_ptr<TensorImpl<T>> grad = nullptr,
			std::shared_ptr<AutoGrad::GradFn<T>> gradFn = nullptr)
			: shape(shape), storage(std::move(storage)), allocator(allocator), offset(offset),
			  requiresGrad(requiresGrad), grad(std::move(grad)), gradFn(std::move(gradFn))
		{}
	};

	/// <summary>
	/// A generic multi-dimensional tensor that stores elements of type T, supports custom allocation, views/slicing, linear and multi-dimensional indexing, iteration, and basic automatic-differentiation (AutoGrad) bookkeeping.
	/// Tensor layout convention: Axis 0 is the batch dimension (if batch exists). All reductions must explicitly decide whether to keep or reduce the batch axis
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensor (e.g., float, double, int).</typeparam>
	template <typename T>
	class Tensor {
	public:
		using Impl = TensorImpl<T>;

		/// <summary>
		/// Constructs a Tensor<T> with the specified shape and allocates its underlying storage using the provided arena allocator.
		/// </summary>
		/// <param name="shape">The tensor shape (dimensions and element count) used to size the underlying storage.</param>
		/// <param name="allocator">The arena allocator used to allocate the tensor's underlying storage. The allocator is referenced to create the storage for the tensor.</param>
		Tensor(const Utils::Shape& shape, Memory::ArenaAllocator& allocator);

		/// <summary>
		/// Constructs a Tensor with the given dimensions and memory allocator by delegating to the constructor that accepts a Utils::Shape.
		/// </summary>
		/// <typeparam name="T">The element type stored in the Tensor.</typeparam>
		/// <param name="dims">An initializer list of sizes for each tensor dimension used to form the tensor shape.</param>
		/// <param name="allocator">Reference to a Memory::ArenaAllocator used for allocating the tensor's memory.</param>
		explicit Tensor(std::initializer_list<size_t> dims, Memory::ArenaAllocator& allocator);

		/// <summary>
		/// Constructs a Tensor<T> from a list of dimension sizes using the provided memory allocator. This constructor delegates to the Tensor constructor that accepts a Utils::Shape.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <param name="dims">A vector of dimension sizes describing the tensor shape.</param>
		/// <param name="allocator">The Memory::ArenaAllocator used to allocate the tensor's storage.</param>
		explicit Tensor(std::vector<size_t> dims, Memory::ArenaAllocator& allocator);

		/// <summary>
		/// Constructs a Tensor<T> from a shared implementation pointer, taking ownership of the provided implementation by moving it into the internal member.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <param name="impl">A std::shared_ptr to the underlying implementation. The pointer is moved into the Tensor's internal implementation member (m_Impl).</param>
		Tensor(std::shared_ptr<Impl> impl);

		/// <summary>
		/// Creates and returns a deep copy of this tensor.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A new Tensor<T> with the same shape and allocator as this tensor, containing an element-wise copy of the original data.</returns>
		Tensor Clone() const;

		/// <summary>
		/// Creates and returns a new Tensor that shares the same underlying storage but is detached from gradient tracking.
		/// </summary>
		/// <typeparam name="T">The element type stored in the Tensor.</typeparam>
		/// <returns>A Tensor<T> that is a shallow copy/view of the original (sharing storage, shape, allocator, and offset) with requiresGrad set to false and its own autograd state. The original tensor is not modified.</returns>
		Tensor Detach() const;

		/// <summary>
		/// Returns a const reference to the tensor's shape without modifying the tensor.
		/// </summary>
		/// <typeparam name="T">The element type stored in the Tensor.</typeparam>
		/// <returns>A const reference to the tensor's Utils::Shape describing its dimensions.</returns>
		const Utils::Shape& GetShape() const;

		/// <summary>
		/// Returns the total number of elements in the tensor.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>The total number of elements (product of the tensor's shape dimensions) as a size_t.</returns>
		size_t NumElements() const;

		/// <summary>
		/// Assigns the given value to every element of the tensor.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <param name="value">The value to assign to each element of the tensor.</param>
		void Fill(const T& value);

		/// <summary>
		/// Returns a pointer to the tensor's underlying element data at the tensor's offset.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A mutable pointer to the first element of this tensor view within the underlying storage (computed as storage.Data() + offset). May be nullptr if the storage is empty.</returns>
		T* Data();

		/// <summary>
		/// Returns a pointer to the tensor's data at its current offset, providing read-only access to the underlying storage.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A pointer to const T pointing at the first element of the tensor's data at the tensor's offset. The pointer refers into the tensor's internal storage and is valid only while the tensor's storage remains unchanged and alive.</returns>
		const T* Data() const;

		/// <summary>
		/// Returns the rank (number of dimensions) of the tensor.
		/// </summary>
		/// <typeparam name="T">The element type stored in the Tensor.</typeparam>
		/// <returns>The number of dimensions (rank) of the tensor as size_t.</returns>
		size_t Rank() const;

		/// <summary>
		/// Returns the tensor's shape dimensions as a const reference to a vector of sizes.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A const reference to a std::vector<size_t> containing the size of each tensor dimension (the shape). The reference refers to internal data owned by the Tensor and must not be modified; its validity is tied to the lifetime of the Tensor.</returns>
		const std::vector<size_t>& Dims() const;

		/// <summary>
		/// Returns a reference to the arena allocator used by this Tensor instance.
		/// </summary>
		/// <typeparam name="T">The element type stored in the Tensor.</typeparam>
		/// <returns>A reference to the Memory::ArenaAllocator held by the tensor's implementation. The reference refers to the allocator owned by the tensor and does not transfer ownership.</returns>
		Memory::ArenaAllocator& GetAllocator();

		/// <summary>
		/// Returns a reference to the arena allocator used by the tensor.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A reference to the tensor's underlying Memory::ArenaAllocator. The reference refers to the allocator owned by the tensor's implementation; do not use it after the tensor is destroyed. Note: although this method is const, it returns a (non-const) reference to the allocator.</returns>
		Memory::ArenaAllocator& GetAllocator() const;

		/// <summary>
		/// Returns a shared pointer to the tensor's underlying implementation.
		/// </summary>
		/// <typeparam name="T">The element type of the Tensor and of the underlying TensorImpl.</typeparam>
		/// <returns>A std::shared_ptr to the TensorImpl<T> backing this Tensor. Ownership is shared; the pointer may be empty (nullptr) if there is no implementation.</returns>
		std::shared_ptr<Impl> GetImpl() const;

		/// <summary>
		/// Returns a pointer to the first element of the tensor's storage, adjusted by the tensor's offset.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A pointer to the tensor's data at the beginning position (storage.Data() + offset). The pointer can be used to iterate over the tensor elements; no bounds checks are performed.</returns>
		T* begin();

		/// <summary>
		/// Returns a pointer to one past the last element in the tensor's contiguous storage.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A T* pointing to the past-the-end element (one past the last valid element) of the tensor's storage.</returns>
		T* end();

		/// <summary>
		/// Returns a read-only pointer to the first element of the tensor's data.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A const pointer to the tensor's first element in the underlying storage (at the tensor's offset). The pointer is read-only and remains valid while the tensor and its underlying storage are not modified or destroyed.</returns>
		const T* begin() const;

		/// <summary>
		/// Returns a const pointer to the element one past the last element of the tensor's data.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A pointer of type const T* pointing to one past the last element in the tensor's internal storage. The pointer is into the tensor's internal data and may be invalidated if the tensor is modified.</returns>
		const T* end() const;

		/// <summary>
		/// Returns a mutable reference to the element at the given linear index in the tensor with bounds checking.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <param name="i">Zero-based linear index of the element to access.</param>
		/// <returns>A mutable reference (T&) to the element at index i. Throws std::out_of_range if i is out of range (i >= NumElements()).</returns>
		T& operator[](size_t i);

		/// <summary>
		/// Returns a const reference to the element at the given linear index, performing bounds checking.
		/// </summary>
		/// <typeparam name="T">The element type stored in the Tensor.</typeparam>
		/// <param name="i">Linear (flattened) index of the element. Must be less than NumElements(); throws std::out_of_range if the index is out of bounds.</param>
		/// <returns>A const reference to the element at index i within the tensor's storage (takes the internal offset into account).</returns>
		const T& operator[](size_t i) const;

		/// <summary>
		/// Return a reference to the tensor element identified by the given multi-dimensional indices. The indices are flattened to compute the element offset and validated; an exception is thrown if out of range.
		/// </summary>
		/// <typeparam name="T">The type of elements stored in the tensor.</typeparam>
		/// <param name="indices">A vector of zero-based indices, one per tensor dimension (its length must match the tensor rank). The indices are flattened to compute the element offset; if the resulting offset is outside the valid range, std::out_of_range is thrown.</param>
		/// <returns>A reference to the element of type T at the specified indices (T&). This allows modifying the element. May throw std::out_of_range on invalid indices.</returns>
		T& operator()(const std::vector<size_t>& indices);

		/// <summary>
		/// Returns a const reference to the tensor element at the specified multi-dimensional indices. Computes a flattened offset from the provided indices, checks bounds, and throws std::out_of_range if the offset is outside the tensor.
		/// </summary>
		/// <typeparam name="T">The element type stored in the Tensor.</typeparam>
		/// <param name="indices">A vector of size_t representing the multi-dimensional indices into the tensor. The vector length should match the tensor's rank; it is flattened to a single offset for element access.</param>
		/// <returns>A const reference to the element of type T at the computed index.</returns>
		const T& operator()(const std::vector<size_t>& indices) const;

		/// <summary>
		/// Returns a mutable reference to the tensor element identified by the provided compile-time integral indices.
		/// </summary>
		/// <typeparam name="Indices">A parameter pack of integral index types; the number of indices must match the tensor's rank and each index is zero-based.</typeparam>
		/// <param name="indices">Zero-based indices for each tensor dimension; the indices are flattened via the tensor's Shape and validated, and std::out_of_range is thrown if any index is out of bounds.</param>
		/// <returns>A mutable reference (T&) to the element at the specified multi-dimensional position, allowing modification of the tensor element.</returns>
		template <typename... Indices, typename = std::enable_if_t<(std::is_integral_v<Indices> && ...)>>
		T& operator()(Indices... indices);

		/// <summary>
		/// Returns a read-only reference to the tensor element identified by the provided compile-time integral indices.
		/// </summary>
		/// <typeparam name="Indices">A parameter pack of integral index types; the number of indices must match the tensor's rank and each index is zero-based.</typeparam>
		/// <param name="indices">Zero-based indices for each tensor dimension; the indices are flattened using the tensor's Shape and validated. Throws std::out_of_range if any index is out of bounds.</param>
		/// <returns>A const reference (const T&) to the element at the specified multi-dimensional position.</returns>
		template <typename... Indices, typename = std::enable_if_t<(std::is_integral_v<Indices> && ...)>>
		const T& operator()(Indices... indices) const;

		/// <summary>
		/// Checks whether this tensor is marked to require gradient computation.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>true if the tensor requires gradients; otherwise false.</returns>
		bool RequiresGrad() const;

		/// <summary>
		/// Checks whether the tensor has an associated gradient.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>true if the tensor has a gradient (m_Impl->grad is not null); otherwise false.</returns>
		bool HasGrad() const;

		/// <summary>
		/// Sets all elements of the tensor's gradient to zero if a gradient buffer exists.
		/// </summary>
		/// <typeparam name="T">The element type of the tensor and its gradient.</typeparam>
		void ZeroGrad();

		/// <summary>
		/// Sets whether the tensor requires gradient computation by updating the internal requiresGrad flag.
		/// </summary>
		/// <typeparam name="T">The element/data type stored in the Tensor.</typeparam>
		/// <param name="require">true to enable gradient tracking for this tensor; false to disable it.</param>
		void SetRequiresGrad(bool require);

		/// <summary>
		/// Returns the gradient tensor for this Tensor. If no gradient is set, returns a new tensor with the same shape filled with zeros.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A Tensor<T> containing the gradient: either the existing gradient or a newly created zero-filled tensor with the same shape and allocator.</returns>
		Tensor<T> Grad();

		/// <summary>
		/// Returns the gradient tensor for this tensor instance.
		/// </summary>
		/// <typeparam name="T">The element type stored in the Tensor.</typeparam>
		/// <returns>A const Tensor<T> representing the gradient. Throws std::runtime_error if the gradient is not available.</returns>
		const Tensor<T> Grad() const;

		/// <summary>
		/// Returns the gradient function associated with this tensor.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <returns>A std::shared_ptr<AutoGrad::GradFn<T>> pointing to the tensor's gradient function. May be nullptr if no gradient function is set.</returns>
		std::shared_ptr<AutoGrad::GradFn<T>> GradFn();

		/// <summary>
		/// Returns the gradient function associated with this tensor without modifying the tensor.
		/// </summary>
		/// <typeparam name="T">The tensor's element/value type; the template parameter used by Tensor and its GradFn.</typeparam>
		/// <returns>A std::shared_ptr to the AutoGrad::GradFn<T> instance for this tensor. The returned pointer may be null if no gradient function is attached.</returns>
		const std::shared_ptr<AutoGrad::GradFn<T>> GradFn() const;

		/// <summary>
		/// Assigns the gradient function for this tensor by storing the provided AutoGrad::GradFn in the tensor's internal implementation.
		/// </summary>
		/// <typeparam name="T">The element type of the Tensor and of the GradFn.</typeparam>
		/// <param name="gradFn">A std::shared_ptr to an AutoGrad::GradFn<T> that will be stored as this tensor's gradient function. The pointer is moved into the tensor's implementation, so the caller's shared_ptr may be left in a moved-from (null) state after the call.</param>
		void SetGradFn(std::shared_ptr<AutoGrad::GradFn<T>> gradFn);

		/// <summary>
		/// Accumulates gradInput into this tensor's stored gradient. If this tensor does not require gradients the function returns immediately. If the internal gradient storage is not yet allocated, it is created and initialized to zero before performing an element-wise addition of gradInput.
		/// </summary>
		/// <typeparam name="T">The element type of the tensor (e.g., float, double, int).</typeparam>
		/// <param name="gradInput">The gradient tensor to add into this tensor's gradient storage. It is expected to have the same number of elements (compatible shape) as this tensor.</param>
		void AccumulateGrad(const Tensor<T>& gradInput);

		/// <summary>
		/// Performs the backward (gradient) pass for this tensor. If requiresGrad is false, the call is a no-op. When called without an explicit gradOutput, this overload only supports scalar tensors (NumElements() == 1); for scalars it constructs a gradOutput tensor filled with ones and forwards to Backward(gradOutput).
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		void Backward();

		/// <summary>
		/// Performs the backward pass for this tensor: if gradients are required, accumulates the provided gradient and invokes the stored gradient function if present.
		/// </summary>
		/// <typeparam name="T">The element type of the tensor (for example, float or double).</typeparam>
		/// <param name="gradOutput">The gradient tensor received from subsequent operations. It is accumulated into this tensor's gradient and forwarded to the saved gradient function if one exists.</param>
		void Backward(const Tensor<T>& gradOutput);

		/// <summary>
		/// Returns a sub-tensor containing the rows in the half-open range [start, end). The result shares the original tensor's underlying storage (no data copy). Throws on invalid rank or range.
		/// </summary>
		/// <typeparam name="T">The tensor element type.</typeparam>
		/// <param name="start">Inclusive 0-based index of the first row to include.</param>
		/// <param name="end">Exclusive 0-based index one past the last row to include. Must satisfy 0 <= start < end <= number of rows.</param>
		/// <returns>A Tensor<T> that views the selected rows (shares storage, adjusted shape and offset).</returns>
		Tensor<T> SliceRows(size_t start, size_t end) const;

		/// <summary>
		/// Concatenates a list of tensors along the first dimension (dimension 0) and returns a new tensor containing the concatenated data. All tensors must be non-empty, have rank > 0, share the same allocator as the first tensor, and have identical sizes for all dimensions except the first. The result uses the allocator of the first tensor and contains a copy of the elements.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensors.</typeparam>
		/// <param name="tensors">A const reference to a vector of Tensor<T> objects to concatenate. Must not be empty. All tensors must have the same rank (> 0), the same allocator as the first tensor, and matching sizes for dimensions 1..rank-1; their dimension 0 sizes are summed to form the result's first dimension.</param>
		/// <returns>A new Tensor<T> whose shape equals the input shape with dimension 0 set to the sum of the input tensors' dimension-0 sizes. The returned tensor contains copies of the input elements in order.</returns>
		static Tensor<T> Concat(const std::vector<Tensor<T>>& tensors);

	private:
		std::shared_ptr<Impl> m_Impl; /// A shared pointer that holds or references the Impl instance associated with this object.
	};
}

#include "tensor.inl"
