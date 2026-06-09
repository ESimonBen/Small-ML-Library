// module.h
#pragma once
#include <mlCore/parameters/parameter.h>

namespace MLCore::NN {
	template <typename T>
	using NamedParameter = std::pair<std::string, std::reference_wrapper<NN::Parameter<T>>>;

	// Class Forwarding for RegisteredModule
	template <typename T>
	class Module;

	template <typename T>
	struct RegisteredModule {
		std::string name;
		std::unique_ptr<Module<T>> module;
	};

	template <typename T>
	class Module {
	public:
		virtual ~Module() = default;

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const = 0;

		void Add(const std::string& name, std::unique_ptr<Module<T>> mod);

		void Add(std::unique_ptr<Module<T>> mod);

		virtual std::vector<std::reference_wrapper<NN::Parameter<T>>> GetParameters();

		virtual std::vector<NamedParameter<T>> GetNamedParameters();

		TensorCore::Tensor<T> operator()(const TensorCore::Tensor<T>& input);

		virtual void Train();

		virtual void Evaluate();

		bool IsTraining() const;

	protected:
		virtual void CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out);

		void CollectSubmoduleParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out);

		virtual void CollectNamedParameters(const std::string& name, std::vector<NamedParameter<T>>& out);

		void CollectNamedSubmoduleParameters(const std::string& name, std::vector<NamedParameter<T>>& out);

	protected:
		/*std::vector<std::unique_ptr<Module<T>>> m_Submodules;*/
		std::vector<RegisteredModule<T>> m_Submodules;
		size_t m_NameCounter = 0;
		bool m_IsTraining = true;
	};
}

#include "module.inl"