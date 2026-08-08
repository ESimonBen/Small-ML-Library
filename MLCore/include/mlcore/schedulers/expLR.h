 /// expLR.h
#pragma once
#include <cmath>
#include "lrScheduler.h"

namespace MLCore::Schedulers {
	/// <summary>
	/// A learning-rate scheduler that applies exponential decay to each parameter group's base learning rate by multiplying it with a running multiplier on each update.
	/// </summary>
	/// <typeparam name="T">Numeric type used for learning rates and internal calculations (e.g., float or double).</typeparam>
	template <typename T>
	class ExponentialLR : public LRScheduler<T> {
	public:
		/// <summary>
		/// Constructs an ExponentialLR learning-rate scheduler that initializes base learning rates from the provided optimizer and sets the decay factor.
		/// </summary>
		/// <param name="opt">Reference to an Optimizers::Optimizer<T> whose parameter groups supply the initial learning rates to be stored as base LRs.</param>
		/// <param name="gamma">Decay factor (multiplier) used to scale the learning rate each step.</param>
		ExponentialLR(Optimizers::Optimizer<T>& opt, T gamma)
			: LRScheduler<T>(opt), m_Gamma(gamma), m_Multiplier(static_cast<T>(1)) {
			std::vector<Optimizers::ParameterGroup<T>>& paramGroups = this->m_Opt.ParamGroups();
			m_BaseLRs.reserve(paramGroups.size());

			for (Optimizers::ParameterGroup<T>& paramGroup : paramGroups) {
				m_BaseLRs.push_back(paramGroup.learningRate);
			}
		}

		/// <summary>
		/// Updates optimizer parameter groups' learning rates by scaling stored base learning rates with an internal multiplier that is decayed by m_Gamma. The method validates that the multiplier and resulting learning rates are finite and positive before applying them.
		/// </summary>
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

		/// <summary>
		/// Returns the type name for the object.
		/// </summary>
		/// <returns>A std::string containing the literal type name "ExponentialLR".</returns>
		virtual std::string TypeName() const override {
			return "ExponentialLR";
		}

		/// <summary>
		/// Serializes the object's internal state into the provided binary writer. It writes m_Gamma, m_Multiplier, then the size and contents of m_BaseLRs and m_LastLRs. This method is const and overrides a base implementation.
		/// </summary>
		/// <param name="writer">Reference to a Serialization::BinaryWriter used to receive the serialized data. The function writes scalar values and then the array sizes followed by their elements.</param>
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

		/// <summary>
		/// Restores the scheduler's state from a binary reader. Reads and validates m_Gamma and m_Multiplier, checks that the number of parameter groups matches the optimizer's ParamGroups(), and reads arrays for m_BaseLRs and m_LastLRs. Throws std::runtime_error on invalid numeric values or parameter-group mismatches.
		/// </summary>
		/// <param name="reader">Serialization::BinaryReader used to read serialized state (gamma, multiplier, counts, and rate arrays).</param>
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
		std::vector<T> m_BaseLRs; /// A container of base LRs for each parameter
		T m_Gamma; /// Decay factor (multiplier) used to scale the learning rate each step.
		T m_Multiplier; /// Constant multiplier rather than using std::pow
	};
}