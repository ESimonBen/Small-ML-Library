/// broadcastingLayers.inl
#include <mlCore/operations/broadcast/broadcast.h>

namespace MLCore::NN {
	template <typename T>
	inline ReshapeLayer<T>::ReshapeLayer(const Utils::Shape& targetShape)
		: m_TargetShape(targetShape)
	{}
	
	template <typename T>
	inline TensorCore::Tensor<T> ReshapeLayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		if (input.Rank() == 1) {
			return Operations::Reshape(input, m_TargetShape);
		}

		size_t targetRank = m_TargetShape.Rank();

		std::vector<size_t> finalDims;
		finalDims.reserve(targetRank + 1); /// Reserve for all target shape dimensions + batch dimension
		finalDims.push_back(input.Dims()[0]); /// Adds the batch dimension in directly

		for (size_t i = 0; i < targetRank; ++i) {
			finalDims.push_back(m_TargetShape[i]);
		}

		return Operations::Reshape(input, Utils::Shape{finalDims});
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> FlattenLayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		return Operations::Flatten(input);
	}
	
	template <typename T>
	inline UnflattenLayer<T>::UnflattenLayer(size_t dim, const Utils::Shape& innerShape)
		: m_Dim(dim), m_InnerShape(innerShape)
	{}
	
	template <typename T>
	inline TensorCore::Tensor<T> UnflattenLayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		return Operations::Unflatten(input, m_Dim, m_InnerShape);
	}
}