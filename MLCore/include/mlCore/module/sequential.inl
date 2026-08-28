 /// sequential.inl
#include <utility>
#include <type_traits>

namespace MLCore::NN {
	template <typename T>
	template <typename ModuleType, typename... Args>
	inline void Sequential<T>::EmplaceNamed(const std::string& name, Args&&... args) {
		static_assert(std::is_base_of_v<Module<T>, ModuleType>, "ModuleType must be of type Module");

		this->Add(name, std::make_unique<ModuleType>(std::forward<Args>(args)...));
	}

	template <typename T>
	template <typename ModuleType, typename... Args>
	inline void Sequential<T>::Emplace(Args&&... args) {
		static_assert(std::is_base_of_v<Module<T>, ModuleType>, "ModuleType must be of type Module");

		this->Add(std::make_unique<ModuleType>(std::forward<Args>(args)...));
	}

	template <typename T>
	inline void Sequential<T>::Add(const std::string& name, std::unique_ptr<Module<T>> mod) {
		Module<T>::Add(name, std::move(mod));
	}
	
	template <typename T>
	inline void Sequential<T>::Add(std::unique_ptr<Module<T>> mod) {
		Module<T>::Add(std::move(mod));
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> Sequential<T>::Forward(const TensorCore::Tensor<T>& input) {
		TensorCore::Tensor<T> inp = input;

		for (const RegisteredModule<T>& layer : this->m_Submodules) {
			inp = layer.module->Forward(inp);
		}

		return inp;
	}
}