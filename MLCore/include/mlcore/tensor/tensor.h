// tensor.h
#pragma once
#include <memory>
#include <vector>
#include <mlCore/utils/shape.h>
#include <mlCore/memory/storage.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::TensorCore {
	template <typename T>
	class Tensor {
	public:
		Tensor(const Utils::Shape& shape, Memory::ArenaAllocator& allocator);
		Tensor(const Tensor& other);
		Tensor(Tensor&& other) noexcept;
		Tensor& operator=(const Tensor& other) noexcept;
		Tensor& operator=(Tensor&& other) noexcept;
		explicit Tensor(std::initializer_list<size_t> dims, Memory::ArenaAllocator& allocator);
		explicit Tensor(std::vector<size_t> dims, Memory::ArenaAllocator& allocator);

		const Utils::Shape& GetShape() const;
		size_t NumElements() const;
		void Fill(const T& value);

		T* Data();
		const T* Data() const;

		size_t Rank() const;
		const std::vector<size_t>& Dims() const;

		Memory::ArenaAllocator& GetAllocator();
		const Memory::ArenaAllocator& GetAllocator() const;

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
		bool IsLeaf() const;
		void SetRequiresGrad(bool require);

		Tensor<T>* Grad();
		const Tensor<T>* Grad() const;
		Tensor<T> Detach();

		AutoGrad::GradFn<T>* GradFn();
		const AutoGrad::GradFn<T>* GradFn() const;

		void SetGradFn(AutoGrad::GradFn<T>* gradFn);
		void AccumulateGrad(const Tensor<T>& gradInput);

		void Backward(const Tensor<T>& gradOutput);
		void Backward(Memory::ArenaAllocator& allocator);

	private:
		Utils::Shape m_Shape;
		Memory::ArenaAllocator* m_Allocator;
		Memory::Storage<T> m_Storage;
		bool m_RequiresGrad = false;
		bool m_Visited = false;
		std::unique_ptr<Tensor<T>> m_Grad = nullptr;
		std::unique_ptr<AutoGrad::GradFn<T>> m_GradFn = nullptr;
	};
}

#include "tensor.inl"