// lrScheduler.h
#pragma once
#include <mlCore/optimizers/optimizer.h>

namespace MLCore::Schedulers {
	template <typename T>
	class LRScheduler {
	public:
		LRScheduler(Optimizers::Optimizer<T>& opt)
			: m_Opt(opt), m_LastLR(opt.LearningRate())
		{}

		virtual ~LRScheduler() = default;

		virtual void UpdateLR() = 0;

	protected:
		Optimizers::Optimizer<T>& m_Opt;
		T m_LastLR;
	};
}