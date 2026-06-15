// lrScheduler.h
#pragma once
#include <mlCore/optimizers/optimizer.h>
#include <mlCore/serialization/binaryArchive.h>

namespace MLCore::Schedulers {
	template <typename T>
	class LRScheduler {
	public:
		LRScheduler(Optimizers::Optimizer<T>& opt)
			: m_Opt(opt) {
			for (Optimizers::ParameterGroup<T>& paramGroup : opt.ParamGroups()) {
				m_LastLRs.push_back(paramGroup.learningRate);
			}
		}

		virtual ~LRScheduler() = default;

		virtual void UpdateLR() = 0;

		virtual void SaveState(Serialization::BinaryWriter& writer) const = 0;
		virtual void LoadState(Serialization::BinaryReader& reader) = 0;

	protected:
		Optimizers::Optimizer<T>& m_Opt;
		std::vector<T> m_LastLRs;
	};
}