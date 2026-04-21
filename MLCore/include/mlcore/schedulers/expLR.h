// expLR.h
#pragma once
#include <cmath>
#include "lrScheduler.h"

namespace MLCore::Schedulers {
	template <typename T>
	class ExponentialLR : public LRScheduler<T> {
	public:
		ExponentialLR(Optimizers::Optimizer<T>& opt, T gamma)
			: LRScheduler<T>(opt), m_Gamma(gamma), m_Step(0) {
			m_BaseLR = opt.LearningRate();
		}

		virtual void UpdateLR() override {
			m_Step++;

			this->m_LastLR = this->m_Opt.LearningRate();
			T lr = m_BaseLR * std::pow(m_Gamma, m_Step);

			if (lr > static_cast<T>(0) && std::isfinite(lr)) {
				this->m_Opt.SetLearningRate(lr);
			}
		}

	private:
		T m_BaseLR;
		T m_Gamma;
		int m_Step;
	};
}