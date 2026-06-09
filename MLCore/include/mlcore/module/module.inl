// module.inl
#include <string>

namespace MLCore::NN {
	template <typename T>
	void Module<T>::Add(const std::string& name, std::unique_ptr<Module<T>> mod) {
		m_Submodules.emplace_back(RegisteredModule<T>{name, std::move(mod)});
	}

	template <typename T>
	void Module<T>::Add(std::unique_ptr<Module<T>> mod) {
		Add("layer" + std::to_string(m_NameCounter++), std::move(mod));
	}

	template <typename T>
	std::vector<std::reference_wrapper<NN::Parameter<T>>> Module<T>::GetParameters(){
		std::vector<std::reference_wrapper<NN::Parameter<T>>> out;
		CollectSubmoduleParameters(out);

		return out;
	}

	template <typename T>
	std::vector<NamedParameter<T>> Module<T>::GetNamedParameters() {
		std::vector<NamedParameter<T>> out;
		CollectNamedSubmoduleParameters("", out);

		return out;
	}

	template <typename T>
	void Module<T>::CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out){
		// Don't need an implementation here
	}

	template <typename T>
	void Module<T>::CollectNamedParameters(const std::string& name, std::vector<NamedParameter<T>>& out) {
		// Don't need an implementation here
	}

	template <typename T>
	void Module<T>::CollectSubmoduleParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) {
		CollectParameters(out);

		for (const RegisteredModule<T>& sub : m_Submodules) {
			sub.module->CollectSubmoduleParameters(out);
		}
	}

	template <typename T>
	void Module<T>::CollectNamedSubmoduleParameters(const std::string& name, std::vector<NamedParameter<T>>& out) {
		CollectNamedParameters(name, out);

		for (RegisteredModule<T>& sub : m_Submodules) {
			std::string childPrefix = (name.empty()) ? sub.name : name + "." + sub.name;

			sub.module->CollectNamedSubmoduleParameters(childPrefix, out);
		}
	}

	template <typename T>
	TensorCore::Tensor<T> Module<T>::operator()(const TensorCore::Tensor<T>& input) {
		return Forward(input);
	}

	template <typename T>
	void Module<T>::Train() {
		m_IsTraining = true;

		for (RegisteredModule<T>& sub : this->m_Submodules) {
			sub.module->Train();
		}
	}

	template <typename T>
	void Module<T>::Evaluate() {
		m_IsTraining = false;

		for (RegisteredModule<T>& sub : this->m_Submodules) {
			sub.module->Evaluate();
		}
	}

	template <typename T>
	bool Module<T>::IsTraining() const {
		return m_IsTraining;
	}
}