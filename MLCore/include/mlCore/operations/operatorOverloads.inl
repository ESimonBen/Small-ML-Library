#include "operatorOverloads.h"
/// operatorOverloads.inl

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> operator+(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		return Add(A, B);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator-(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		return Subtract(A, B);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator*(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		return Multiply(A, B);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator/(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		return Divide(A, B);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator+(const TensorCore::Tensor<T>& Input, T Scalar) {
		return AddScalar(Input, Scalar);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator+(T Scalar, const TensorCore::Tensor<T>& Input) {
		return AddScalar(Input, Scalar);
	}
	
	template<typename T>
	inline TensorCore::Tensor<T> operator++(TensorCore::Tensor<T>& Input) {
		Input = AddScalar(Input, static_cast<T>(1));
		return Input;
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator++(TensorCore::Tensor<T>& Input, int) {
		TensorCore::Tensor<T> Temp = Input;
		++Input;
		return Temp;
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator-(const TensorCore::Tensor<T>& Input, T Scalar) {
		return SubtractScalar(Input, Scalar, false);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator-(T Scalar, const TensorCore::Tensor<T>& Input) {
		return SubtractScalar(Input, Scalar, true);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator--(TensorCore::Tensor<T>& Input) {
		Input = SubtractScalar(Input, static_cast<T>(1), false);
		return Input;
	}

	template <typename T>
	inline TensorCore::Tensor<T> operator--(TensorCore::Tensor<T>& Input, int) {
		TensorCore::Tensor<T> Temp = Input;
		++Input;
		return Temp;
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator*(const TensorCore::Tensor<T>& Input, T Scalar) {
		return MultiplyScalar(Input, Scalar);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator*(T Scalar, const TensorCore::Tensor<T>& Input) {
		return MultiplyScalar(Input, Scalar);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator/(const TensorCore::Tensor<T>& Input, T Scalar) {
		return DivideScalar(Input, Scalar, false);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> operator/(T Scalar, const TensorCore::Tensor<T>& Input) {
		return DivideScalar(Input, Scalar, true);
	}
	
	template<typename T>
	inline TensorCore::Tensor<T> operator==(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		return Equal(A, B);
	}
	
	template<typename T>
	inline TensorCore::Tensor<T> operator!=(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		return NotEqual(A, B);
	}
	
	template<typename T>
	inline TensorCore::Tensor<T> operator>(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		return GreaterThan(A, B);
	}
	
	template<typename T>
	inline TensorCore::Tensor<T> operator<(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		return LessThan(A, B);
	}
	
	template<typename T>
	inline TensorCore::Tensor<T> operator>=(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		return GreaterThanOrEqual(A, B);
	}
	
	template<typename T>
	inline TensorCore::Tensor<T> operator<=(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		return LessThanOrEqual(A, B);
	}
}