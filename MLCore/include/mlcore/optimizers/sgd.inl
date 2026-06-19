 /// sgd.inl
#include "sgd.h"

namespace MLCore::Optimizers {
	template <typename T>
	SGD<T>::SGD(std::vector<std::reference_wrapper<NN::Parameter<T>>> params, T learningRate, T weightDecay)
		: Optimizer<T>(params, learningRate, weightDecay)
	{}
	
	template <typename T>
	SGD<T>::SGD(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay)
		: Optimizer<T>(params, learningRate, weightDecay)
	{}
	
	template <typename T>
	SGD<T>::SGD(std::vector<ParameterGroup<T>> groups)
		: Optimizer<T>(groups)
	{}
	
	template <typename T>
	void SGD<T>::Step() {
		this->ClipGradients();

		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			T learningRate = paramGroup.learningRate;
			T weightDecay = paramGroup.weightDecay;

			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				TensorCore::Tensor<T>& param = p.Data();

				if (!param.RequiresGrad() || !param.HasGrad()) {
					continue;
				}

				TensorCore::Tensor<T> grad = param.Grad();
				size_t size = param.NumElements();

				for (size_t i = 0; i < size; ++i) {
					T gradScalar = grad[i];

					if (weightDecay != static_cast<T>(0)) {
						gradScalar += weightDecay * param[i];
					}

					param[i] -= learningRate * gradScalar;
				}
			}
		}
	}
	
	template<typename T>
	std::string SGD<T>::TypeName() const {
		return "SGD";
	}
	
	template <typename T>
	void SGD<T>::SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const {
		size_t numGroups = this->m_ParamGroups.size();
		writer.Write(numGroups);

		for (ParameterGroup<T>& group : this->m_ParamGroups) {
			writer.Write(group.learningRate);
			writer.Write(group.weightDecay);
		}
	}
	
	template <typename T>
	void SGD<T>::LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) {
		size_t numGroups;
		reader.Read(numGroups);

		if (numGroups != this->m_ParamGroups.size()) {
			throw std::runtime_error("ERROR: Optimizer parameter group mismatch");
		}

		for (ParameterGroup<T>& group : this->m_ParamGroups) {
			reader.Read(group.learningRate);
			reader.Read(group.weightDecay);
		}
	}
	
	template <typename T>
	SGDMomentum<T>::SGDMomentum(std::vector<std::reference_wrapper<NN::Parameter<T>>> params, T learningRate, T momentum, T weightDecay, T dampening, bool nesterov)
		: Optimizer<T>(params, learningRate, weightDecay), m_Momentum(momentum), m_Dampening(dampening), m_Nesterov(nesterov) {
			for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
				for (auto& ref : paramGroup.params) {
					NN::Parameter<T>& p = ref.get();
					NN::ParamID paramID = p.id;
					TensorCore::Tensor<T>& param = p.Data();

					TensorCore::Tensor<T> velocity{ param.GetShape(), param.GetAllocator() };
					velocity.Fill(static_cast<T>(0));
					m_Velocities.try_emplace(paramID, velocity);
				}
			}
	}
	
	template <typename T>
	SGDMomentum<T>::SGDMomentum(std::vector<NN::Parameter<T>>& params, T learningRate, T momentum, T weightDecay, T dampening, bool nesterov)
		: Optimizer<T>(params, learningRate, weightDecay), m_Momentum(momentum), m_Dampening(dampening), m_Nesterov(nesterov) {
			for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
				for (auto& ref : paramGroup.params) {
					NN::Parameter<T>& p = ref.get();
					NN::ParamID paramID = p.id;
					TensorCore::Tensor<T>& param = p.Data();

					TensorCore::Tensor<T> velocity{ param.GetShape(), param.GetAllocator() };
					velocity.Fill(static_cast<T>(0));
					m_Velocities.try_emplace(paramID, velocity);
				}
			}
	}
	
	template <typename T>
	SGDMomentum<T>::SGDMomentum(std::vector<ParameterGroup<T>> groups, T momentum, T dampening, bool nesterov)
		: Optimizer<T>(groups), m_Momentum(momentum), m_Dampening(dampening), m_Nesterov(nesterov) {
		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				TensorCore::Tensor<T> velocity{ param.GetShape(), param.GetAllocator() };
				velocity.Fill(static_cast<T>(0));
				m_Velocities.try_emplace(paramID, velocity);
			}
		}
	}
	
	template <typename T>
	void SGDMomentum<T>::Step() {
		this->ClipGradients();

		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			T learningRate = paramGroup.learningRate;
			T weightDecay = paramGroup.weightDecay;

			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				auto velocityIt = m_Velocities.find(paramID);

				if (velocityIt == m_Velocities.end()) {
					throw std::runtime_error("ERROR: Step: Missing optimizer state");
				}

				TensorCore::Tensor<T>& velocity = velocityIt->second;

				if (!param.RequiresGrad() || !param.HasGrad()) {
					continue;
				}

				TensorCore::Tensor<T> grad = param.Grad();

				size_t size = param.NumElements();
				for (size_t j = 0; j < size; ++j) {
					T gradScalar = grad[j];

					// Weight decay
					if (weightDecay != static_cast<T>(0)) {
						gradScalar += weightDecay * param[j];
					}

					T scaledGrad = (static_cast<T>(1) - m_Dampening) * gradScalar;

					// Momentum with dampening
					velocity[j] = (m_Momentum * velocity[j]) + scaledGrad;

					// Nesterov or standard stochastic gradient descent
					if (m_Nesterov) {
						param[j] -= learningRate * (scaledGrad + m_Momentum * velocity[j]);
					}
					else {
						param[j] -= learningRate * velocity[j];
					}
				}
			}
		}
	}
	
	template <typename T>
	std::string SGDMomentum<T>::TypeName() const {
		return "SGDMomentum";
	}
	
	template <typename T>
	void SGDMomentum<T>::SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const {
		auto namedParams = model.GetNamedParameters();
		std::unordered_map<NN::ParamID, std::string> idToName;

		for (const auto& [name, param] : namedParams) {
			idToName[param.get().id] = name;
		}

		writer.Write(m_Momentum);
		writer.Write(m_Dampening);
		writer.Write(m_Nesterov);

		size_t numGroups = this->m_ParamGroups.size();
		writer.Write(numGroups);

		for (const ParameterGroup<T>& group : this->m_ParamGroups) {
			writer.Write(group.learningRate);
			writer.Write(group.weightDecay);

			size_t numParams = group.params.size();
			writer.Write(numParams);

			for (const std::reference_wrapper<NN::Parameter<T>>& ref : group.params) {
				const NN::Parameter<T>& param = ref.get();

				const std::string& name = idToName.at(param.id);
				const size_t nameLength = name.size();
				writer.Write(nameLength);
				writer.WriteArray(name.data(), nameLength);

				auto velocityIter = m_Velocities.find(param.id);

				if (velocityIter == m_Velocities.end()) {
					throw std::runtime_error("ERROR: Velocity not found");
				}

				const TensorCore::Tensor<T>& velocity = velocityIter->second;
				writer.WriteTensor(velocity);
			}
		}
	}
	
	template <typename T>
	void SGDMomentum<T>::LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) {
		auto namedParams = model.GetNamedParameters();
		std::unordered_map<std::string, NN::Parameter<T>*> nameToParam;

		for (auto& [name, param] : namedParams) {
			nameToParam[name] = &param.get();
		}

		reader.Read(m_Momentum);
		reader.Read(m_Dampening);
		reader.Read(m_Nesterov);

		size_t numGroups;
		reader.Read(numGroups);

		if (numGroups != this->m_ParamGroups.size()) {
			throw std::runtime_error("ERROR: Optimizer parameter group mismatch");
		}

		for (size_t i = 0; i < numGroups; ++i) {
			ParameterGroup<T>& paramGroup = this->m_ParamGroups[i];

			reader.Read(paramGroup.learningRate);
			reader.Read(paramGroup.weightDecay);

			size_t numParams;
			reader.Read(numParams);

			if (numParams != paramGroup.params.size()) {
				throw std::runtime_error("ERROR: Parameter group size mismatch");
			}

			for (size_t j = 0; j < numParams; ++j) {
				size_t nameLength;
				reader.Read(nameLength);

				std::string name(nameLength, '\0');
				reader.ReadArray(name.data(), nameLength);

				auto paramIt = nameToParam.find(name);

				if (paramIt == nameToParam.end()) {
					throw std::runtime_error("ERROR: Optimizer parameter '" + name + "' not found");
				}

				NN::ParamID paramID = paramIt->second->id;

				auto velocityIt = m_Velocities.find(paramID);

				if (velocityIt == m_Velocities.end()) {
					throw std::runtime_error("ERROR: Optimizer velocity not found");
				}

				TensorCore::Tensor<T>& velocity = velocityIt->second;
				reader.ReadTensor(velocity);
			}
		}
	}
}