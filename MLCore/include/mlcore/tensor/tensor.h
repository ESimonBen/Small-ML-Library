// tensor.h
#pragma once
#include <memory>
#include <vector>
#include <mlCore/utils/shape.h>
#include <mlCore/memory/storage.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::TensorCore {
	// Tensor implementation (used in computation graph)
	template <typename T>
	struct TensorImpl {
		Utils::Shape shape;
		Memory::Storage<T> storage;
		Memory::ArenaAllocator* allocator;
		bool requiresGrad;
		std::shared_ptr<TensorImpl<T>> grad;
		std::shared_ptr<AutoGrad::GradFn<T>> gradFn;

		TensorImpl(const Utils::Shape& shape,
			Memory::Storage<T> storage,
			Memory::ArenaAllocator* allocator,
			bool requiresGrad = false,
			std::shared_ptr<TensorImpl<T>> grad = nullptr,
			std::shared_ptr<AutoGrad::GradFn<T>> gradFn = nullptr)
			: shape(shape),
			storage(std::move(storage)),
			allocator(allocator),
			requiresGrad(requiresGrad),
			grad(std::move(grad)),
			gradFn(std::move(gradFn)) 
		{}
	};

	// Tensor wrapper (gives acces to actual tensor node while not BEING the node)
	template <typename T>
	class Tensor {
	public:
		using Impl = TensorImpl<T>;

		Tensor(const Utils::Shape& shape, Memory::ArenaAllocator& allocator);
		explicit Tensor(std::initializer_list<size_t> dims, Memory::ArenaAllocator& allocator);
		explicit Tensor(std::vector<size_t> dims, Memory::ArenaAllocator& allocator);

		// New
		Tensor(std::shared_ptr<Impl> impl);

		Tensor Clone() const;
		Tensor Detach() const; // For performance's sake (viewing instead of creating)

		const Utils::Shape& GetShape() const;
		size_t NumElements() const;
		void Fill(const T& value);

		T* Data();
		const T* Data() const;

		size_t Rank() const;
		const std::vector<size_t>& Dims() const;

		Memory::ArenaAllocator& GetAllocator();
		std::shared_ptr<Impl> GetImpl() const;

		// Added these to use iterators (that's the reason for the difference in style from the rest of the code)
		T* begin();
		T* end();

		const T* begin() const;
		const T* end() const;

		// For Linear Data Access
		T& operator[](size_t i);
		const T& operator[](size_t i) const;

		// For Multi-Dimensional Data Access
		T& operator()(const std::vector<size_t>& indices);
		const T& operator()(const std::vector<size_t>& indices) const;

		template <typename... Indices, typename = std::enable_if_t<(std::is_integral_v<Indices> && ...)>>
		T& operator()(Indices... indices);

		template <typename... Indices, typename = std::enable_if_t<(std::is_integral_v<Indices> && ...)>>
		const T& operator()(Indices... indices) const;

		//AutoGrad API
		bool RequiresGrad() const;
		bool HasGrad() const;
		void ZeroGrad();
		void SetRequiresGrad(bool require);

		Tensor<T> Grad();
		const Tensor<T> Grad() const;

		std::shared_ptr<AutoGrad::GradFn<T>> GradFn();
		const std::shared_ptr<AutoGrad::GradFn<T>> GradFn() const;

		void SetGradFn(std::shared_ptr<AutoGrad::GradFn<T>> gradFn);
		void AccumulateGrad(const Tensor<T>& gradInput);

		void Backward();
		void Backward(const Tensor<T>& gradOutput);

		// Static fillers
		// static Tensor Zero();

	private:
		std::shared_ptr<Impl> m_Impl;
	};
}

#include "tensor.inl"