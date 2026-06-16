// expLR.h
#pragma once
#include <cmath>
#include "lrScheduler.h"

namespace MLCore::Schedulers {
	template <typename T>
	class ExponentialLR : public LRScheduler<T> {
	public:
		ExponentialLR(Optimizers::Optimizer<T>& opt, T gamma)
			: LRScheduler<T>(opt), m_Gamma(gamma), m_Multiplier(static_cast<T>(1)) {
			std::vector<Optimizers::ParameterGroup<T>>& paramGroups = this->m_Opt.ParamGroups();
			m_BaseLRs.reserve(paramGroups.size());

			for (Optimizers::ParameterGroup<T>& paramGroup : paramGroups) {
				m_BaseLRs.push_back(paramGroup.learningRate);
			}
		}

		virtual void UpdateLR() override {
			m_Multiplier *= m_Gamma;

			if (!std::isfinite(m_Multiplier)) {
				m_Multiplier = static_cast<T>(0);
			}

			std::vector<Optimizers::ParameterGroup<T>>& paramGroups = this->m_Opt.ParamGroups();
			size_t size = paramGroups.size();

			assert(m_BaseLRs.size() == size);

			for (size_t i = 0; i < size; ++i) {
				T lr = m_BaseLRs[i] * m_Multiplier;

				if (std::isfinite(lr) && lr > static_cast<T>(0)) {
					paramGroups[i].learningRate = lr;
				}
			}
		}

		virtual std::string TypeName() const override {
			return "ExponentialLR";
		}

		virtual void SaveState(Serialization::BinaryWriter& writer) const override {
			writer.Write(m_Gamma);
			writer.Write(m_Multiplier);

			size_t n = m_BaseLRs.size();
			writer.Write(n);
			writer.WriteArray(m_BaseLRs.data(), n);

			size_t numLRs = this->m_LastLRs.size();

			writer.Write(numLRs);
			writer.WriteArray(this->m_LastLRs.data(), numLRs);
		}

		virtual void LoadState(Serialization::BinaryReader& reader) override {
			reader.Read(m_Gamma);
			reader.Read(m_Multiplier);

			if (!std::isfinite(m_Gamma) || m_Gamma <= static_cast<T>(0)) {
				throw std::runtime_error("ERROR: Invalid ExponentialLR gamma");
			}

			if (!std::isfinite(m_Multiplier) || m_Multiplier < static_cast<T>(0)) {
				throw std::runtime_error("ERROR: Invalid ExponentialLR multiplier");
			}

			size_t n;
			reader.Read(n);
			if (n != this->m_Opt.ParamGroups().size()) {
				throw std::runtime_error("ERROR: Scheduler parameter group mismatch");
			}

			m_BaseLRs.resize(n);
			reader.ReadArray(m_BaseLRs.data(), n);

			size_t numLRs;
			reader.Read(numLRs);

			if (numLRs != this->m_Opt.ParamGroups().size()) {
				throw std::runtime_error("ERROR: Scheduler parameter group mismatch");
			}

			this->m_LastLRs.resize(numLRs);
			reader.ReadArray(this->m_LastLRs.data(), numLRs);
		}

	private:
		std::vector<T> m_BaseLRs;
		T m_Gamma;
		T m_Multiplier; // Constant multiplication rather than using std::pow
	};
}