// stepLR.h
#pragma once
#include "lrScheduler.h"

namespace MLCore::Schedulers {
	template <typename T>
	class StepLR : public LRScheduler<T> {
	public:
		StepLR(Optimizers::Optimizer<T>& opt, int stepSize, T gamma)
			: LRScheduler<T>(opt), m_StepSize(stepSize), m_Gamma(gamma), m_Step(0)
		{}

		virtual void UpdateLR() override {
			// LR is a piecewise constant
			if (++m_Step >= m_StepSize) {
				std::vector<Optimizers::ParameterGroup<T>>& paramGroups = this->m_Opt.ParamGroups();
				size_t size = paramGroups.size();

				for (size_t i = 0; i < size; ++i) {
					auto& lr = paramGroups[i].learningRate;

					this->m_LastLRs[i] = lr;

					lr *= m_Gamma;

					if (!std::isfinite(lr)) {
						lr = static_cast<T>(1e-8); // Safety clamp if the learning rate is infinite
					}
					else if (lr <= static_cast<T>(0)){
						lr = static_cast<T>(1e-12); // Safety clamp if the learning rate is not a positive value
					}
				}

				m_Step = 0;
			}
		}

		virtual void SaveState(Serialization::BinaryWriter& writer) const override {
			writer.Write(m_StepSize);
			writer.Write(m_Gamma);
			writer.Write(m_Step);
		}

		virtual void LoadState(Serialization::BinaryReader& reader) override {
			reader.Read(m_StepSize);
			reader.Read(m_Gamma);
			reader.Read(m_Step);

			if (m_StepSize <= 0) {
				throw std::runtime_error("ERROR: Invalid StepLR step size");
			}

			if (!std::isfinite(m_Gamma) || m_Gamma <= static_cast<T>(0)) {
				throw std::runtime_error("ERROR: Invalid StepLR gamma");
			}

			if (m_Step < 0 || m_Step >= m_StepSize) {
				throw std::runtime_error("ERROR: Invalid StepLR step count");
			}
		}

	private:
		int m_StepSize;
		T m_Gamma;
		int m_Step;
	};
}