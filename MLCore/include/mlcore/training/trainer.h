// trainer.h
#pragma once
#include <functional>
#include <mlCore/tensor/tensor.h>
#include <mlCore/module/module.h>
#include <mlCore/optimizers/optimizer.h>
//#include <mlCore/operations/loss/loss.h>

namespace MLCore::Training {
	template <typename T>
	using LossFn = std::function<TensorCore::Tensor<T>(const TensorCore::Tensor<T>&, const TensorCore::Tensor<T>&/*, Operations::Reduction, Memory::ArenaAllocator&*/)>;

	template <typename T>
	class Trainer {
	public:
		Trainer(NN::Module<T>& model, Optimizers::Optimizer<T>& optimizer, LossFn<T> lossFn);

		void Fit(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, int epochs);

		// Optional hook for debugging
		std::function<void(int epoch, T loss)> OnEpochEnd;

	private:
		NN::Module<T>& m_Model;
		Optimizers::Optimizer<T>& m_Optimizer;
		LossFn<T> m_LossFn;
	};
}

#include "trainer.inl"