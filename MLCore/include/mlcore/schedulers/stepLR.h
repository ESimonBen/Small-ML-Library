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
				this->m_LastLR = this->m_Opt.LearningRate();
				T lr = this->m_Opt.LearningRate() * m_Gamma;

				if (lr > static_cast<T>(0) && std::isfinite(lr)) {
					this->m_Opt.SetLearningRate(lr);
				}
				m_Step = 0;
			}
		}

	private:
		int m_StepSize;
		T m_Gamma;
		int m_Step;
	};
}