 /// stepLR.h
#pragma once
#include "lrScheduler.h"

namespace MLCore::Schedulers {
	/// <summary>
	/// A step-wise learning rate scheduler that multiplies each parameter group's learning rate by a factor (gamma) every stepSize calls to UpdateLR.
	/// </summary>
	/// <typeparam name="T">Numeric type used for learning rates and gamma (e.g., float or double).</typeparam>
	template <typename T>
	class StepLR : public LRScheduler<T> {
	public:
		/// <summary>
		/// Constructs a StepLR learning-rate scheduler that decays the optimizer's learning rate every stepSize steps by multiplying it with gamma.
		/// </summary>
		/// <param name="opt">Reference to the optimizer whose learning rate will be scheduled.</param>
		/// <param name="stepSize">Number of steps between each learning-rate decay.</param>
		/// <param name="gamma">Multiplicative decay factor applied to the learning rate at each step.</param>
		StepLR(Optimizers::Optimizer<T>& opt, int stepSize, T gamma)
			: LRScheduler<T>(opt), m_StepSize(stepSize), m_Gamma(gamma), m_Step(0)
		{}

		/// <summary>
		/// Updates learning rates using a piecewise-constant schedule. Increments an internal step counter and, when it reaches m_StepSize, multiplies each parameter group's learning rate by m_Gamma, stores the previous values in m_LastLRs, clamps non-finite learning rates to 1e-8 and non-positive values to 1e-12, and resets the step counter.
		/// </summary>
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
						lr = static_cast<T>(1e-8); /// Safety clamp if the learning rate is infinite
					}
					else if (lr <= static_cast<T>(0)){
						lr = static_cast<T>(1e-12); /// Safety clamp if the learning rate is not a positive value
					}
				}

				m_Step = 0;
			}
		}

		/// <summary>
		/// Returns the runtime type name for this object. This method overrides the base implementation and does not modify object state.
		/// </summary>
		/// <returns>A std::string containing the type name "StepLR".</returns>
		virtual std::string TypeName() const override {
			return "StepLR";
		}

		/// <summary>
		/// Serializes the object's internal state to a binary writer.
		/// </summary>
		/// <param name="writer">Reference to a Serialization::BinaryWriter used to write the object's state (m_StepSize, m_Gamma, m_Step, the count of last LRs, and the array of last LRs).</param>
		virtual void SaveState(Serialization::BinaryWriter& writer) const override {
			writer.Write(m_StepSize);
			writer.Write(m_Gamma);
			writer.Write(m_Step);

			size_t numLRs = this->m_LastLRs.size();

			writer.Write(numLRs);
			writer.WriteArray(this->m_LastLRs.data(), numLRs);
		}

		/// <summary>
		/// Loads the scheduler state from a binary reader, restoring internal fields (m_StepSize, m_Gamma, m_Step, and m_LastLRs), and validates the loaded values. Throws std::runtime_error on invalid or mismatched data.
		/// </summary>
		/// <param name="reader">Reference to a Serialization::BinaryReader used to read the persisted scheduler state. The method reads m_StepSize, m_Gamma, m_Step, the number of learning-rate groups, and the array of last learning rates, and performs validation checks (throwing std::runtime_error on failure).</param>
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

			size_t numLRs;
			reader.Read(numLRs);

			if (numLRs != this->m_Opt.ParamGroups().size()) {
				throw std::runtime_error("ERROR: Scheduler parameter group mismatch");
			}

			this->m_LastLRs.resize(numLRs);
			reader.ReadArray(this->m_LastLRs.data(), numLRs);
		}

	private:
		int m_StepSize; /// Number of steps between each LR decay
		T m_Gamma; /// Multiplicative decay factor applied to the learning rate at each step.
		int m_Step; /// An integer that stores the current step
	};
}