 /// lrScheduler.h
#pragma once
#include <mlCore/optimizers/optimizer.h>
#include <mlCore/serialization/binaryArchive.h>

namespace MLCore::Schedulers {
	/// <summary>
	/// Abstract base class for learning-rate schedulers that track and update an optimizer's per-parameter-group learning rates.
	/// </summary>
	/// <typeparam name="T">Numeric type used for learning rates (for example, float or double).</typeparam>
	template <typename T>
	class LRScheduler {
	public:
		/// <summary>
		/// Constructs an LRScheduler for the given optimizer, stores a reference to the optimizer, and initializes m_LastLRs with each parameter group's learning rate.
		/// </summary>
		/// <param name="opt">Reference to an Optimizers::Optimizer<T> whose parameter groups' learningRate values are read and stored.</param>
		LRScheduler(Optimizers::Optimizer<T>& opt)
			: m_Opt(opt) {
			for (Optimizers::ParameterGroup<T>& paramGroup : opt.ParamGroups()) {
				m_LastLRs.push_back(paramGroup.learningRate);
			}
		}

		/// <summary>
		/// Virtual default destructor for LRScheduler that ensures proper cleanup in derived classes.
		/// </summary>
		virtual ~LRScheduler() = default;

		/// <summary>
		/// Pure virtual method that updates LR; must be implemented by derived classes.
		/// </summary>
		virtual void UpdateLR() = 0;

		/// <summary>
		/// Pure virtual member function that returns the object's type name.
		/// </summary>
		/// <returns>A std::string containing the type name. The method is const and pure virtual, so derived classes must override it.</returns>
		virtual std::string TypeName() const = 0;

		/// <summary>
		/// Serializes the object's state to the provided binary writer. This is a pure virtual function that must be implemented by derived classes and does not modify the observable state of the object.
		/// </summary>
		/// <param name="writer">Reference to a Serialization::BinaryWriter that will receive the serialized representation of the object's state.</param>
		virtual void SaveState(Serialization::BinaryWriter& writer) const = 0;

		/// <summary>
		/// Loads the object's state from the given binary reader. This is a pure virtual method that must be implemented by derived classes.
		/// </summary>
		/// <param name="reader">The Serialization::BinaryReader to read serialized state from.</param>
		virtual void LoadState(Serialization::BinaryReader& reader) = 0;

		/// <summary>
		/// Returns a const reference to the stored vector of last LRs (getter).
		/// </summary>
		/// <returns>A const reference to the internal std::vector<T> m_LastLRs. The reference is read-only and remains valid while the object (and its member m_LastLRs) exists.</returns>
		const std::vector<T>& GetLastLRs() const {
			return m_LastLRs;
		}

	protected:
		Optimizers::Optimizer<T>& m_Opt; /// A reference to an Optimizer specialized with type T.
		std::vector<T> m_LastLRs; /// A container of the previous learning rates.
	};
}