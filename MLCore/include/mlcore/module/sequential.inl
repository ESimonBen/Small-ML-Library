// sequential.inl
#include <utility>
#include <type_traits>

namespace MLCore::NN {
	template <typename T>
	template <typename ModuleType, typename... Args>
	void Sequential<T>::Emplace(Args&&... args) {
		this->Add(std::make_unique<ModuleType>(std::forward<Args>(args)...));
	}

	template <typename T>
	void Sequential<T>::Add(std::unique_ptr<Module<T>> mod) {
		Module<T>::Add(std::move(mod));
	}

	template <typename T>
	TensorCore::Tensor<T> Sequential<T>::Forward(const TensorCore::Tensor<T>& input) const {
		TensorCore::Tensor<T> inp = input;

		for (const std::unique_ptr<Module<T>>& layer : this->m_Submodules) {
			inp = layer->Forward(inp);
		}

		return inp;
	}

	template <typename T>
	void Sequential<T>::Train() {
		this->m_IsTraining = true;

		for (const std::unique_ptr<Module<T>>& layer : this->m_Submodules) {
			layer->Train();
		}
	}

	template <typename T>
	void Sequential<T>::Evaluate() {
		this->m_IsTraining = false;

		for (const std::unique_ptr<Module<T>>& layer : this->m_Submodules) {
			layer->Evaluate();
		}
	}
}