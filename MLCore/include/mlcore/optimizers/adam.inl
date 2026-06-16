// adam.inl
#include <cmath>
#include <unordered_map>
#include <mlCore/serialization/binaryArchive.h>
#include "adam.h"

namespace MLCore::Optimizers {
	template <typename T>
	Adam<T>::Adam(std::vector <std::reference_wrapper<NN::Parameter<T>>>& params, T learningRate, T weightDecay, T beta1, T beta2, T epsilon)
		: Optimizer<T>(params, learningRate, weightDecay), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		m_Epsilon(epsilon), m_Timestep(0) {
			for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
				for (auto& ref : paramGroup.params) {
					NN::Parameter<T>& p = ref.get();
					NN::ParamID paramID = p.id;
					TensorCore::Tensor<T>& param = p.Data();

					TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
					TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

					m.Fill(static_cast<T>(0));
					v.Fill(static_cast<T>(0));

					m_FirstMoment.try_emplace(paramID, m);
					m_SecondMoment.try_emplace(paramID, v);
				}
			}
	}

	template <typename T>
	Adam<T>::Adam(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay, T beta1, T beta2, T epsilon)
		: Optimizer<T>(params, learningRate, weightDecay), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		  m_Epsilon(epsilon), m_Timestep(0) {
			  for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
				  for (auto& ref : paramGroup.params) {
					  NN::Parameter<T>& p = ref.get();
					  NN::ParamID paramID = p.id;
					  TensorCore::Tensor<T>& param = p.Data();

					  TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
					  TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

					  m.Fill(static_cast<T>(0));
					  v.Fill(static_cast<T>(0));

					  m_FirstMoment.try_emplace(paramID, m);
					  m_SecondMoment.try_emplace(paramID, v);
				  }
			  }
	}

	template <typename T>
	Adam<T>::Adam(std::vector<ParameterGroup<T>> groups, T beta1, T beta2, T epsilon)
		: Optimizer<T>(groups), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		m_Epsilon(epsilon), m_Timestep(0) {
		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
				TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

				m.Fill(static_cast<T>(0));
				v.Fill(static_cast<T>(0));

				m_FirstMoment.try_emplace(paramID, m);
				m_SecondMoment.try_emplace(paramID, v);
			}
		}
	}

	template <typename T>
	void Adam<T>::Step() {
		m_Timestep++;
		this->ClipGradients();

		m_BetaPow1 *= m_Beta1;
		m_BetaPow2 *= m_Beta2;
		T bias1 = (static_cast<T>(1) - m_BetaPow1);
		T bias2 = (static_cast<T>(1) - m_BetaPow2);

		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			T learningRate = paramGroup.learningRate;
			T weightDecay = paramGroup.weightDecay;

			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				auto mIt = m_FirstMoment.find(paramID);
				auto vIt = m_SecondMoment.find(paramID);

				if (mIt == m_FirstMoment.end() || vIt == m_SecondMoment.end()) {
					throw std::runtime_error("ERROR: Step: Missing optimizer state");
				}

				TensorCore::Tensor<T>& m = mIt->second;
				TensorCore::Tensor<T>& v = vIt->second;


				if (!param.RequiresGrad() || !param.HasGrad()) {
					continue;
				}

				if (m.NumElements() == 0) {
					m = TensorCore::Tensor<T>{ param.GetShape(), param.GetAllocator() };
					m.Fill(static_cast<T>(0));
				}

				if (v.NumElements() == 0) {
					v = TensorCore::Tensor<T>{ param.GetShape(), param.GetAllocator() };
					v.Fill(static_cast<T>(0));
				}

				TensorCore::Tensor<T> grad = param.Grad();

				size_t size = param.NumElements();

				for (size_t i = 0; i < size; ++i) {
					T gradScalar = grad[i];

					// Weight decay
					if (weightDecay != static_cast<T>(0)) {
						gradScalar += weightDecay * param[i];
					}

					// Update biased moments
					m[i] = m_Beta1 * m[i] + (static_cast<T>(1) - m_Beta1) * gradScalar;
					v[i] = m_Beta2 * v[i] + (static_cast<T>(1) - m_Beta2) * (gradScalar * gradScalar);

					// Bias correction
					T m_hat = m[i] / bias1;
					T v_hat = v[i] / bias2;

					param[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + m_Epsilon));
				}
			}
		}
	}

	template<typename T>
	std::string Adam<T>::TypeName() const {
		return "Adam";
	}

	template <typename T>
	void Adam<T>::SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const {
		auto namedParams = model.GetNamedParameters();
		std::unordered_map<NN::ParamID, std::string> idToName;

		for (const auto& [name, param] : namedParams) {
			idToName[param.get().id] = name;
		}

		writer.Write(m_Beta1);
		writer.Write(m_Beta2);
		writer.Write(m_BetaPow1);
		writer.Write(m_BetaPow2);
		writer.Write(m_Epsilon);
		writer.Write(m_Timestep);

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

				auto firstMomIter = m_FirstMoment.find(param.id);

				if (firstMomIter == m_FirstMoment.end()) {
					throw std::runtime_error("ERROR: First moment not found");
				}

				auto secMomIter = m_SecondMoment.find(param.id);

				if (secMomIter == m_SecondMoment.end()) {
					throw std::runtime_error("ERROR: Second moment not found");
				}

				const TensorCore::Tensor<T>& firstMoment = firstMomIter->second;
				const TensorCore::Tensor<T>& secondMoment = secMomIter->second;

				writer.WriteTensor(firstMoment);
				writer.WriteTensor(secondMoment);
			}
		}
	}

	template <typename T>
	void Adam<T>::LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) {
		auto namedParams = model.GetNamedParameters();
		std::unordered_map<std::string, NN::Parameter<T>*> nameToParam;

		for (auto& [name, param] : namedParams) {
			nameToParam[name] = &param.get();
		}

		reader.Read(m_Beta1);
		reader.Read(m_Beta2);
		reader.Read(m_BetaPow1);
		reader.Read(m_BetaPow2);
		reader.Read(m_Epsilon);
		reader.Read(m_Timestep);

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

				auto firstMomIter = m_FirstMoment.find(paramID);

				if (firstMomIter == m_FirstMoment.end()) {
					throw std::runtime_error("ERROR: First moment not found");
				}

				auto secMomIter = m_SecondMoment.find(paramID);

				if (firstMomIter == m_FirstMoment.end()) {
					throw std::runtime_error("ERROR: Second moment not found");
				}

				TensorCore::Tensor<T>& firstMoment = firstMomIter->second;
				TensorCore::Tensor<T>& secondMoment = secMomIter->second;

				reader.ReadTensor(firstMoment);
				reader.ReadTensor(secondMoment);
			}
		}
	}

	template <typename T>
	AdamW<T>::AdamW(std::vector <std::reference_wrapper<NN::Parameter<T>>>& params, T learningRate, T weightDecay, T beta1, T beta2, T epsilon)
		: Optimizer<T>(params, learningRate, weightDecay), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		m_Epsilon(epsilon), m_Timestep(0) {
		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
				TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

				m.Fill(static_cast<T>(0));
				v.Fill(static_cast<T>(0));

				m_FirstMoment.try_emplace(paramID, m);
				m_SecondMoment.try_emplace(paramID, v);
			}
		}
	}

	template <typename T>
	AdamW<T>::AdamW(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay, T beta1, T beta2, T epsilon)
		: Optimizer<T>(params, learningRate, weightDecay), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		  m_Epsilon(epsilon), m_Timestep(0) {
			for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
				for (auto& ref : paramGroup.params) {
					NN::Parameter<T>& p = ref.get();
					NN::ParamID paramID = p.id;
					TensorCore::Tensor<T>& param = p.Data();

					TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
					TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

					m.Fill(static_cast<T>(0));
					v.Fill(static_cast<T>(0));

					m_FirstMoment.try_emplace(paramID, m);
					m_SecondMoment.try_emplace(paramID, v);
				}
			}
	}

	template <typename T>
	AdamW<T>::AdamW(std::vector<ParameterGroup<T>> groups, T beta1, T beta2, T epsilon)
		: Optimizer<T>(groups), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		m_Epsilon(epsilon), m_Timestep(0) {
		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
				TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

				m.Fill(static_cast<T>(0));
				v.Fill(static_cast<T>(0));

				m_FirstMoment.try_emplace(paramID, m);
				m_SecondMoment.try_emplace(paramID, v);
			}
		}
	}

	template <typename T>
	void AdamW<T>::Step() {
		m_Timestep++;
		this->ClipGradients();

		m_BetaPow1 *= m_Beta1;
		m_BetaPow2 *= m_Beta2;
		T bias1 = (static_cast<T>(1) - m_BetaPow1);
		T bias2 = (static_cast<T>(1) - m_BetaPow2);

		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			T learningRate = paramGroup.learningRate;
			T weightDecay = paramGroup.weightDecay;

			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				auto mIt = m_FirstMoment.find(paramID);
				auto vIt = m_SecondMoment.find(paramID);

				if (mIt == m_FirstMoment.end() || vIt == m_SecondMoment.end()) {
					throw std::runtime_error("ERROR: Step: Missing optimizer state");
				}

				TensorCore::Tensor<T>& m = mIt->second;
				TensorCore::Tensor<T>& v = vIt->second;


				if (!param.RequiresGrad() || !param.HasGrad()) {
					continue;
				}

				if (m.NumElements() == 0) {
					m = TensorCore::Tensor<T>{ param.GetShape(), param.GetAllocator() };
					m.Fill(static_cast<T>(0));
				}

				if (v.NumElements() == 0) {
					v = TensorCore::Tensor<T>{ param.GetShape(), param.GetAllocator() };
					v.Fill(static_cast<T>(0));
				}

				TensorCore::Tensor<T> grad = param.Grad();

				size_t size = param.NumElements();

				for (size_t i = 0; i < size; ++i) {
					T gradScalar = grad[i];

					// Weight decay
					if (weightDecay != static_cast<T>(0)) {
						param[i] -= learningRate * weightDecay * param[i];
					}

					// Update biased moments
					m[i] = m_Beta1 * m[i] + (static_cast<T>(1) - m_Beta1) * gradScalar;
					v[i] = m_Beta2 * v[i] + (static_cast<T>(1) - m_Beta2) * (gradScalar * gradScalar);

					// Bias correction
					T m_hat = m[i] / bias1;
					T v_hat = v[i] / bias2;

					param[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + m_Epsilon));
				}
			}
		}
	}

	template<typename T>
	std::string AdamW<T>::TypeName() const {
		return "AdamW";
	}

	template <typename T>
	void AdamW<T>::SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const {
		auto namedParams = model.GetNamedParameters();
		std::unordered_map<NN::ParamID, std::string> idToName;

		for (const auto& [name, param] : namedParams) {
			idToName[param.get().id] = name;
		}

		writer.Write(m_Beta1);
		writer.Write(m_Beta2);
		writer.Write(m_BetaPow1);
		writer.Write(m_BetaPow2);
		writer.Write(m_Epsilon);
		writer.Write(m_Timestep);

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

				auto firstMomIter = m_FirstMoment.find(param.id);

				if (firstMomIter == m_FirstMoment.end()) {
					throw std::runtime_error("ERROR: First moment not found");
				}

				auto secMomIter = m_SecondMoment.find(param.id);

				if (firstMomIter == m_FirstMoment.end()) {
					throw std::runtime_error("ERROR: Second moment not found");
				}

				const TensorCore::Tensor<T>& firstMoment = firstMomIter->second;
				const TensorCore::Tensor<T>& secondMoment = secMomIter->second;

				writer.WriteTensor(firstMoment);
				writer.WriteTensor(secondMoment);
			}
		}
	}

	template <typename T>
	void AdamW<T>::LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) {
		auto namedParams = model.GetNamedParameters();
		std::unordered_map<std::string, NN::Parameter<T>*> nameToParam;

		for (auto& [name, param] : namedParams) {
			nameToParam[name] = &param.get();
		}

		reader.Read(m_Beta1);
		reader.Read(m_Beta2);
		reader.Read(m_BetaPow1);
		reader.Read(m_BetaPow2);
		reader.Read(m_Epsilon);
		reader.Read(m_Timestep);

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

				auto firstMomIter = m_FirstMoment.find(paramID);

				if (firstMomIter == m_FirstMoment.end()) {
					throw std::runtime_error("ERROR: First moment not found");
				}

				auto secMomIter = m_SecondMoment.find(paramID);

				if (secMomIter == m_SecondMoment.end()) {
					throw std::runtime_error("ERROR: Second moment not found");
				}

				TensorCore::Tensor<T>& firstMoment = firstMomIter->second;
				TensorCore::Tensor<T>& secondMoment = secMomIter->second;

				reader.ReadTensor(firstMoment);
				reader.ReadTensor(secondMoment);
			}
		}
	}
}