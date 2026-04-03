// tensor.h
#pragma once
#include <vector>
#include <type_traits>
#include <mlCore/memory/storage.h>
#include <mlCore/utils/shape.h>

namespace MLCore::TensorCore {
	template <typename T>
	class Tensor {
	public:
		Tensor(const Utils::Shape& shape, Memory::ArenaAllocator& allocator);
		explicit Tensor(std::initializer_list<size_t> dims, Memory::ArenaAllocator& allocator);

		const Utils::Shape& GetShape() const;

		size_t NumElements() const;

		void Fill(const T& value);

		T* Data();
		const T* Data() const;

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

	private:
		Utils::Shape m_Shape;
		Memory::Storage<T> m_Storage;
	};
}

#include "tensor.inl"