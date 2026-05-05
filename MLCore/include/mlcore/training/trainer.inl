// trainer.inl

namespace MLCore::Training {
	template <typename T>
	Trainer<T>::Trainer(NN::Module<T>& model, Optimizers::Optimizer<T>& optimizer, LossFn<T> lossFn)
		: m_Model(model), m_Optimizer(optimizer), m_LossFn(lossFn)
	{}

	template <typename T>
	void Trainer<T>::Fit(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, int epochs) {
		for (int epoch = 0; epoch < epochs; ++epoch) {
			 // Forward propogation
			auto pred = m_Model(inputs);

			// Backpropogation
			auto loss = m_LossFn(pred, targets);

			m_Optimizer.ZeroGrad();
			loss.Backward();
			m_Optimizer.Step();

			if (OnEpochEnd) {
				OnEpochEnd(epoch, loss[0]);
			}
		}
	}
}