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

	private:
		std::vector<T> m_BaseLRs;
		T m_Gamma;
		T m_Multiplier; // Constant multiplication rather than using std::pow
	};
}