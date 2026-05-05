// loss.inl
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <mlCore/operations/operations.h>

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> MeanSquaredError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: MeanSquaredError: Tensor size mismatch");
		}

		if (axis >= predictions.Rank()) {
			throw std::out_of_range("ERROR: MeanSquaredError: Axis out of bounds");
		}

		TensorCore::Tensor<T> diff = Operations::Subtract(targets, predictions, allocator);
		TensorCore::Tensor<T> square = Operations::Square(diff, allocator);
		TensorCore::Tensor<T> perSample = Operations::AxisMean(square, axis, allocator, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return Operations::MeanAll(perSample, allocator);
		}

		case Reduction::Sum:
		{
			return Operations::SumAll(perSample, allocator);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}

	template <typename T>
	inline TensorCore::Tensor<T> MeanAbsoluteError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: MeanAbsoluteError: Tensor size mismatch");
		}

		if (axis >= predictions.Rank()) {
			throw std::out_of_range("ERROR: MeanAbsoluteError: Axis out of bounds");
		}

		TensorCore::Tensor<T> diff = Operations::Subtract(targets, predictions, allocator);
		TensorCore::Tensor<T> abs = Operations::Abs(diff, allocator);
		TensorCore::Tensor<T> perSample = Operations::AxisMean(abs, axis, allocator, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return Operations::MeanAll(perSample, allocator);
		}

		case Reduction::Sum:
		{
			return Operations::SumAll(perSample, allocator);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}

	template <typename T>
	inline TensorCore::Tensor<T> BinaryCrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: BinaryCrossEntropy: Tensor size mismatch");
		}

		if (axis >= predictions.Rank()) {
			throw std::out_of_range("ERROR: BinaryCrossEntropy: Axis out of bounds");
		}

		const T epsilon = static_cast<T>(1e-7);

		TensorCore::Tensor<T> clamp = Operations::Clamp(predictions, epsilon, static_cast<T>(1) - epsilon, allocator);
		TensorCore::Tensor<T> logP = Operations::Log(clamp, allocator);
		TensorCore::Tensor<T> term1 = Operations::Multiply(targets, logP, allocator);

		TensorCore::Tensor<T> oneMinusT = Operations::SubtractScalar(targets, static_cast<T>(1), allocator, true);
		TensorCore::Tensor<T> oneMinusP = Operations::SubtractScalar(clamp, static_cast<T>(1), allocator, true);
		TensorCore::Tensor<T> logOneMinusP = Operations::Log(oneMinusP, allocator);
		TensorCore::Tensor<T> term2 = Operations::Multiply(oneMinusT, logOneMinusP, allocator);

		TensorCore::Tensor<T> addition = Operations::Add(term1, term2, allocator);
		TensorCore::Tensor<T> loss = Operations::Negate(addition, allocator);

		TensorCore::Tensor<T> perSample = Operations::AxisMean(loss, axis, allocator, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return Operations::MeanAll(perSample, allocator);
		}

		case Reduction::Sum:
		{
			return Operations::SumAll(perSample, allocator);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}

	template <typename T>
	TensorCore::Tensor<T> BinaryCrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator) {
		if (logits.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: BinaryCrossEntropyWithLogits: Tensor shape mismatch");
		}

		if (axis >= logits.Rank()) {
			throw std::out_of_range("ERROR: BinaryCrossEntropyWithLogits: Axis out of bounds");
		}

		TensorCore::Tensor<T> max = Operations::ReLU(logits, allocator);
		TensorCore::Tensor<T> abs = Operations::Abs(logits, allocator);
		TensorCore::Tensor<T> negateAbs = Operations::Negate(abs, allocator);
		TensorCore::Tensor<T> exp = Operations::Exp(negateAbs, allocator);
		TensorCore::Tensor<T> sum = Operations::AddScalar(exp, static_cast<T>(1), allocator);
		TensorCore::Tensor<T> term1 = Operations::Log(sum, allocator);

		TensorCore::Tensor<T> mul = Operations::Multiply(logits, targets, allocator);
		TensorCore::Tensor<T> sub = Operations::Subtract(max, mul, allocator);

		TensorCore::Tensor<T> loss = Operations::Add(sub, term1, allocator);

		TensorCore::Tensor<T> perSample = Operations::AxisMean(loss, axis, allocator, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return Operations::MeanAll(perSample, allocator);
		}

		case Reduction::Sum:
		{
			return Operations::SumAll(perSample, allocator);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}

	// This assumes that the result of Softmax is being passed in
	// (Softmax returns the probabilities of each element of the prediction)
	template <typename T>
	inline TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: CrossEntropy: Tensor shape mismatch");
		}

		if (axis >= predictions.Rank()) {
			throw std::out_of_range("ERROR: CrossEntropy: Axis out of bounds");
		}

		const T epsilon = static_cast<T>(1e-7);

		TensorCore::Tensor<T> clamp = Operations::Clamp(predictions, epsilon, static_cast<T>(1) - epsilon, allocator);
		TensorCore::Tensor<T> logClamp = Operations::Log(clamp, allocator);

		TensorCore::Tensor<T> negate = Operations::Negate(targets, allocator);
		TensorCore::Tensor<T> loss = Operations::Multiply(negate, logClamp, allocator);

		TensorCore::Tensor<T> perSample = Operations::AxisMean(loss, axis, allocator, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return Operations::MeanAll(perSample, allocator);
		}

		case Reduction::Sum:
		{
			return Operations::SumAll(perSample, allocator);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}

	template <typename T>
	TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator) {
		if (logits.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: CrossEntropyWithLogits: Tensor shape mismatch");
		}

		if (axis >= logits.Rank()) {
			throw std::out_of_range("ERROR: CrossEntropyWithLogits: Axis out of bounds");
		}

		TensorCore::Tensor<T> logSoftmax = Operations::AxisLogSoftmax(logits, axis, allocator);

		TensorCore::Tensor<T> mul = Operations::Multiply(targets, logSoftmax, allocator);
		TensorCore::Tensor<T> neg = Operations::Negate(mul, allocator);

		TensorCore::Tensor<T> perSample = Operations::AxisMean(neg, axis, allocator, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return Operations::MeanAll(perSample, allocator);
		}

		case Reduction::Sum:
		{
			return Operations::SumAll(perSample, allocator);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}

	// Assuming axis = last axis here

	template <typename T>
	TensorCore::Tensor<T> MeanSquaredError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator) {
		return MeanSquaredError(predictions, targets, predictions.Rank() - 1, config, allocator);
	}

	template <typename T>
	TensorCore::Tensor<T> MeanAbsoluteError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator) {
		return MeanAbsoluteError(predictions, targets, predictions.Rank() - 1, config, allocator);
	}

	template <typename T>
	TensorCore::Tensor<T> BinaryCrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator) {
		return BinaryCrossEntropy(predictions, targets, predictions.Rank() - 1, config, allocator);
	}

	template <typename T>
	TensorCore::Tensor<T> BinaryCrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator) {
		return BinaryCrossEntropyWithLogits(logits, targets, logits.Rank() - 1, config, allocator);
	}

	template <typename T>
	TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator) {
		return CrossEntropy(predictions, targets, predictions.Rank() - 1, config, allocator);
	}

	template <typename T>
	TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator) {
		return CrossEntropyWithLogits(logits, targets, logits.Rank() - 1, config, allocator);
	}
}
