// loss.inl
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <mlCore/operations/operations.h>

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> MeanSquaredError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: MeanSquaredError: Tensor size mismatch");
		}

		size_t axis = predictions.Rank() - 1;
		TensorCore::Tensor<T> diff = Operations::Subtract(targets, predictions, allocator);
		TensorCore::Tensor<T> square = Operations::Square(diff, allocator);
		TensorCore::Tensor<T> perSample = Operations::AxisMean(square, axis, allocator);
		TensorCore::Tensor<T> result = Operations::Mean(perSample, allocator);

		if (predictions.RequiresGrad()) {
			result.SetRequiresGrad(true);
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MeanAbsoluteError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: MeanAbsoluteError: Tensor size mismatch");
		}

		size_t axis = predictions.Rank() - 1;

		TensorCore::Tensor<T> diff = Operations::Subtract(targets, predictions, allocator);
		TensorCore::Tensor<T> abs = Operations::Abs(diff, allocator);
		TensorCore::Tensor<T> perSample = Operations::AxisMean(abs, axis, allocator);
		TensorCore::Tensor<T> result = Operations::Mean(perSample, allocator);

		if (predictions.RequiresGrad()) {
			result.SetRequiresGrad(true);
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> BinaryCrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: BinaryCrossEntropy: Tensor size mismatch");
		}

		const T epsilon = static_cast<T>(1e-7);

		TensorCore::Tensor<T> clamp = Operations::Clamp(predictions, epsilon, static_cast<T>(1) - epsilon, allocator);
		TensorCore::Tensor<T> logP = Operations::Log(predictions, allocator);
		TensorCore::Tensor<T> term1 = Operations::Multiply(clamp, logP, allocator);

		TensorCore::Tensor<T> oneMinusT = Operations::SubtractScalar(targets, static_cast<T>(1), allocator, true);
		TensorCore::Tensor<T> oneMinusP = Operations::SubtractScalar(predictions, static_cast<T>(1), allocator, true);
		TensorCore::Tensor<T> logOneMinusP = Operations::Log(oneMinusP, allocator);
		TensorCore::Tensor<T> term2 = Operations::Multiply(oneMinusT, logOneMinusP, allocator);

		TensorCore::Tensor<T> addition = Operations::Add(term1, term2, allocator);
		TensorCore::Tensor<T> result = Operations::Negate(addition, allocator);

		if (predictions.RequiresGrad()) {
			result.SetRequiresGrad(true);
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> BinaryCrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (logits.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: BinaryCrossEntropyWithLogits: Tensor shape mismatch");
		}

		TensorCore::Tensor<T> max = Operations::ReLU(logits, allocator);
		TensorCore::Tensor<T> abs = Operations::Abs(max, allocator);
		TensorCore::Tensor<T> negateAbs = Operations::Negate(abs, allocator);
		TensorCore::Tensor<T> exp = Operations::Exp(negateAbs, allocator);
		TensorCore::Tensor<T> sum = Operations::AddScalar(exp, static_cast<T>(1), allocator);
		TensorCore::Tensor<T> term1 = Operations::Log(sum, allocator);

		TensorCore::Tensor<T> mul = Operations::Multiply(logits, targets, allocator);
		TensorCore::Tensor<T> sub = Operations::Subtract(max, mul, allocator);

		TensorCore::Tensor<T> loss = Operations::Add(sub, term1, allocator);

		size_t axis = logits.Rank() - 1;
		TensorCore::Tensor<T> perSample = Operations::AxisMean(loss, axis, allocator);
		TensorCore::Tensor<T> result = Operations::Mean(perSample, allocator);

		if (logits.RequiresGrad()) {
			result.SetRequiresGrad(true);
		}

		return result;
	}

	// This assumes that the result of Softmax is being passed in
	// (Softmax returns the probabilities of each element of the prediction)
	template <typename T>
	inline TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: CrossEntropy: Tensor shape mismatch");
		}

		size_t axis = predictions.Rank() - 1;
		const T epsilon = static_cast<T>(1e-7);

		TensorCore::Tensor<T> clamp = Operations::Clamp(predictions, epsilon, static_cast<T>(1) - epsilon, allocator);
		TensorCore::Tensor<T> logClamp = Operations::Log(clamp, allocator);

		TensorCore::Tensor<T> negate = Operations::Negate(targets, allocator);
		TensorCore::Tensor<T> loss = Operations::Multiply(negate, logClamp, allocator);

		TensorCore::Tensor<T> perSample = Operations::AxisMean(loss, axis, allocator);
		TensorCore::Tensor<T> result = Operations::Mean(perSample, allocator);

		if (predictions.RequiresGrad()) {
			result.SetRequiresGrad(true);
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, size_t axis, Memory::ArenaAllocator& allocator) {
		if (logits.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: CrossEntropyWithLogits: Tensor shape mismatch");
		}

		if (axis >= logits.Rank()) {
			throw std::out_of_range("ERROR: AxisSum: Axis out of bounds");
		}

		TensorCore::Tensor<T> axisSoftmax = Operations::AxisSoftmax(logits, axis, allocator);
		TensorCore::Tensor<T> logSoftmax = Operations::Log(axisSoftmax, allocator); // May cause possible overflow, but I don't care currently

		TensorCore::Tensor<T> mul = Operations::Multiply(targets, logSoftmax, allocator);
		TensorCore::Tensor<T> neg = Operations::Negate(mul, allocator);

		TensorCore::Tensor<T> perSample = Operations::AxisMean(neg, axis, allocator);
		TensorCore::Tensor<T> result = Operations::Mean(perSample, allocator);

		if (logits.RequiresGrad()) {
			result.SetRequiresGrad(true);
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		return CrossEntropyWithLogits(logits, targets, logits.Rank() - 1, allocator);
	}
}