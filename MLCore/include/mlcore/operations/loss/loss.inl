 /// loss.inl
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <mlCore/operations/operations.h>

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> MeanSquaredError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config) {
		if (&predictions.GetAllocator() != &targets.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (predictions.Dims().empty() || targets.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensors cannot be null");
		}

		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: MeanSquaredError: Tensor size mismatch");
		}

		if (axis >= predictions.Rank()) {
			throw std::out_of_range("ERROR: MeanSquaredError: Axis out of bounds");
		}

		TensorCore::Tensor<T> diff = Subtract(targets, predictions);
		TensorCore::Tensor<T> square = Square(diff);
		TensorCore::Tensor<T> perSample = AxisMean(square, axis, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return MeanAll(perSample);
		}

		case Reduction::Sum:
		{
			return SumAll(perSample);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> MeanAbsoluteError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config) {
		if (&predictions.GetAllocator() != &targets.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (predictions.Dims().empty() || targets.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensors cannot be null");
		}

		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: MeanAbsoluteError: Tensor size mismatch");
		}

		if (axis >= predictions.Rank()) {
			throw std::out_of_range("ERROR: MeanAbsoluteError: Axis out of bounds");
		}

		TensorCore::Tensor<T> diff = Subtract(targets, predictions);
		TensorCore::Tensor<T> abs = Abs(diff);
		TensorCore::Tensor<T> perSample = AxisMean(abs, axis, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return MeanAll(perSample);
		}

		case Reduction::Sum:
		{
			return SumAll(perSample);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> BinaryCrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config) {
		if (&predictions.GetAllocator() != &targets.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (predictions.Dims().empty() || targets.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensors cannot be null");
		}

		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: BinaryCrossEntropy: Tensor size mismatch");
		}

		if (axis >= predictions.Rank()) {
			throw std::out_of_range("ERROR: BinaryCrossEntropy: Axis out of bounds");
		}

		const T epsilon = static_cast<T>(1e-7);

		/// term1 = targets * ln(clamp(preds, 1e-7, (1 - 1e-7)))
		TensorCore::Tensor<T> clamp = Clamp(predictions, epsilon, static_cast<T>(1) - epsilon);
		TensorCore::Tensor<T> logP = Log(clamp);
		TensorCore::Tensor<T> term1 = Multiply(targets, logP);

		/// term2 = (1 - targets) * ln(1 - preds)
		TensorCore::Tensor<T> oneMinusT = SubtractScalar(targets, static_cast<T>(1), true);
		TensorCore::Tensor<T> oneMinusP = SubtractScalar(clamp, static_cast<T>(1), true);
		TensorCore::Tensor<T> logOneMinusP = Log(oneMinusP);
		TensorCore::Tensor<T> term2 = Multiply(oneMinusT, logOneMinusP);

		TensorCore::Tensor<T> addition = Add(term1, term2);
		TensorCore::Tensor<T> loss = Negate(addition);

		TensorCore::Tensor<T> perSample = AxisMean(loss, axis, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return MeanAll(perSample);
		}

		case Reduction::Sum:
		{
			return SumAll(perSample);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> BinaryCrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config) {
		if (&logits.GetAllocator() != &targets.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (logits.Dims().empty() || targets.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensors cannot be null");
		}
		
		if (logits.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: BinaryCrossEntropyWithLogits: Tensor shape mismatch");
		}

		if (axis >= logits.Rank()) {
			throw std::out_of_range("ERROR: BinaryCrossEntropyWithLogits: Axis out of bounds");
		}

		TensorCore::Tensor<T> max = ReLU(logits);
		TensorCore::Tensor<T> abs = Abs(logits);
		TensorCore::Tensor<T> negateAbs = Negate(abs);
		TensorCore::Tensor<T> exp = Exp(negateAbs);
		TensorCore::Tensor<T> sum = AddScalar(exp, static_cast<T>(1));
		TensorCore::Tensor<T> term1 = Log(sum);

		TensorCore::Tensor<T> mul = Multiply(logits, targets);
		TensorCore::Tensor<T> sub = Subtract(max, mul);

		TensorCore::Tensor<T> loss = Add(sub, term1);

		TensorCore::Tensor<T> perSample = AxisMean(loss, axis, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return MeanAll(perSample);
		}

		case Reduction::Sum:
		{
			return SumAll(perSample);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config) {
		if (&predictions.GetAllocator() != &targets.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (predictions.Dims().empty() || targets.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensors cannot be null");
		}

		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: CrossEntropy: Tensor shape mismatch");
		}

		if (axis >= predictions.Rank()) {
			throw std::out_of_range("ERROR: CrossEntropy: Axis out of bounds");
		}

		const T epsilon = static_cast<T>(1e-7);

		TensorCore::Tensor<T> clamp = Clamp(predictions, epsilon, static_cast<T>(1) - epsilon);
		TensorCore::Tensor<T> logClamp = Log(clamp);

		TensorCore::Tensor<T> negate = Negate(targets);
		TensorCore::Tensor<T> loss = Multiply(negate, logClamp);

		TensorCore::Tensor<T> perSample = AxisMean(loss, axis, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return MeanAll(perSample);
		}

		case Reduction::Sum:
		{
			return SumAll(perSample);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config) {
		if (&logits.GetAllocator() != &targets.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (logits.Dims().empty() || targets.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensors cannot be null");
		}

		if (logits.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: CrossEntropyWithLogits: Tensor shape mismatch");
		}

		if (axis >= logits.Rank()) {
			throw std::out_of_range("ERROR: CrossEntropyWithLogits: Axis out of bounds");
		}

		TensorCore::Tensor<T> logSoftmax = AxisLogSoftmax(logits, axis);

		TensorCore::Tensor<T> mul = Multiply(targets, logSoftmax);
		TensorCore::Tensor<T> neg = Negate(mul);

		TensorCore::Tensor<T> perSample = AxisMean(neg, axis, true);

		switch (config) {
		case Reduction::None:
		{
			return perSample;
		}
		case Reduction::Mean:
		{
			return MeanAll(perSample);
		}

		case Reduction::Sum:
		{
			return SumAll(perSample);
		}

		default:
			throw std::runtime_error("ERROR: Invalid reduction option/type");
		}
	}

	/// Assuming axis = last axis here

	template <typename T>
	inline TensorCore::Tensor<T> MeanSquaredError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config) {
		return MeanSquaredError(predictions, targets, predictions.Rank() - 1, config);
	}

	template <typename T>
	inline TensorCore::Tensor<T> MeanAbsoluteError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config) {
		return MeanAbsoluteError(predictions, targets, predictions.Rank() - 1, config);
	}

	template <typename T>
	inline TensorCore::Tensor<T> BinaryCrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config) {
		return BinaryCrossEntropy(predictions, targets, predictions.Rank() - 1, config);
	}

	template <typename T>
	inline TensorCore::Tensor<T> BinaryCrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Reduction config) {
		return BinaryCrossEntropyWithLogits(logits, targets, logits.Rank() - 1, config);
	}

	template <typename T>
	inline TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config) {
		return CrossEntropy(predictions, targets, predictions.Rank() - 1, config);
	}

	template <typename T>
	inline TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Reduction config) {
		return CrossEntropyWithLogits(logits, targets, logits.Rank() - 1, config);
	}
}
