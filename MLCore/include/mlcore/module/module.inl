// module.inl

namespace MLCore::NN {
	template <typename T>
	void Module<T>::Add(std::unique_ptr<Module<T>> mod) {
		m_Submodules.push_back(std::move(mod));
	}

	template <typename T>
	std::vector<std::reference_wrapper<NN::Parameter<T>>> Module<T>::GetParameters(){
		std::vector<std::reference_wrapper<NN::Parameter<T>>> out;
		CollectSubmoduleParameters(out);

		return out;
	}

	template <typename T>
	void Module<T>::CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out){
		// Don't need an implementation here
	}

	template <typename T>
	void Module<T>::CollectSubmoduleParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) {
		CollectParameters(out);

		for (const std::unique_ptr<Module<T>>& sub : m_Submodules) {
			sub->CollectSubmoduleParameters(out);
		}
	}

	template <typename T>
	TensorCore::Tensor<T> Module<T>::operator()(const TensorCore::Tensor<T>& input) {
		return Forward(input);
	}
}