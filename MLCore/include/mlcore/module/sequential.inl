// sequential.inl

namespace MLCore::NN {
	template <typename T>
	void Sequential<T>::Add(std::shared_ptr<Module<T>> mod) {
		Module<T>::Add(mod);
	}

	template <typename T>
	TensorCore::Tensor<T> Sequential<T>::Forward(const TensorCore::Tensor<T>& input) {
		TensorCore::Tensor<T> inp = input;

		for (std::shared_ptr<Module<T>>& layer : this->m_Submodules) {
			inp = layer->Forward(inp);
		}

		return inp;
	}
}