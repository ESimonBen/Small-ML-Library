// sequential.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {
	template <typename T>
	class Sequential : public Module<T> {
	public:
		Sequential() = default;

		/*template <typename... Modules>
		Sequential(Modules&&... mods);*/

		template <typename ModuleType, typename... Args>
		void Emplace(Args&&... args);

		void Add(std::unique_ptr<Module<T>> mod);

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;
	};
}

#include "sequential.inl"