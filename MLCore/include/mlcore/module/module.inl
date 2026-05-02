// module.inl

namespace MLCore::NN {
	template <typename T>
	void Module<T>::Add(std::shared_ptr<Module<T>> mod) {
		m_Submodules.push_back(std::move(mod));
	}

	template <typename T>
	std::vector<std::reference_wrapper<NN::Parameter<T>>> Module<T>::GetParameters() const {
		std::vector<std::reference_wrapper<NN::Parameter<T>>> out;
		CollectParameters(out);

		return out;
	}

	template <typename T>
	void Module<T>::CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) const {
		// Don't need an implementation here
	}

	template <typename T>
	void Module<T>::CollectSubmoduleParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) const {
		CollectParameters(out);

		for (const std::shared_ptr<Module<T>>& sub : m_Submodules) {
			sub->CollectSubmoduleParameters(out);
		}
	}
}