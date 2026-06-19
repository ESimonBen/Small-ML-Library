 /// sequential.inl
#include <utility>
#include <type_traits>

namespace MLCore::NN {
	template <typename T>
	template <typename ModuleType, typename... Args>
	void Sequential<T>::EmplaceNamed(const std::string& name, Args&&... args) {
		this->Add(name, std::make_unique<ModuleType>(std::forward<Args>(args)...));
	}

	template <typename T>
	template <typename ModuleType, typename... Args>
	void Sequential<T>::Emplace(Args&&... args) {
		this->Add(std::make_unique<ModuleType>(std::forward<Args>(args)...));
	}

	template <typename T>
	void Sequential<T>::Add(const std::string& name, std::unique_ptr<Module<T>> mod) {
		Module<T>::Add(name, std::move(mod));
	}
	
	template <typename T>
	void Sequential<T>::Add(std::unique_ptr<Module<T>> mod) {
		Module<T>::Add(std::move(mod));
	}
	
	template <typename T>
	TensorCore::Tensor<T> Sequential<T>::Forward(const TensorCore::Tensor<T>& input) const {
		TensorCore::Tensor<T> inp = input;

		for (const RegisteredModule<T>& layer : this->m_Submodules) {
			inp = layer.module->Forward(inp);
		}

		return inp;
	}
}