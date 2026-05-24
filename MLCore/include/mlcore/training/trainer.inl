// trainer.inl

namespace MLCore::Training {
	template <typename T>
	Trainer<T>::Trainer(NN::Module<T>& model, Optimizers::Optimizer<T>& optimizer, LossFn<T> lossFn)
		: m_Model(model), m_Optimizer(optimizer), m_LossFn(lossFn)
	{}

	template <typename T>
	void Trainer<T>::Fit(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets, int epochs, size_t batchSize) {
		size_t numSamples = inputs.Dims()[0];

		for (int epoch = 0; epoch < epochs; ++epoch) {
			T epochLoss = 0;
			size_t batchCount = 0;

			for (size_t i = 0; i < numSamples; i += batchSize) {
				size_t end = std::min(i + batchSize, numSamples);

				TensorCore::Tensor<T> batchX = inputs.SliceRows(i, end);
				TensorCore::Tensor<T> batchY = targets.SliceRows(i, end);

				// Forward propogation
				auto pred = m_Model.Forward(batchX);

				// Loss
				auto loss = m_LossFn(pred, batchY);

				// Backward
				m_Optimizer.ZeroGrad();
				loss.Backward();
				m_Optimizer.Step();

				epochLoss += loss[0];
				batchCount++;

				if (OnEpochEval) {
					OnEpochEval(epoch, pred, batchY);
				}
			}

			epochLoss /= batchCount;

			if (OnEpochEnd) {
				TensorCore::Tensor<T> avgLoss{ {1}, inputs.GetAllocator() };
				avgLoss[0] = epochLoss;
				OnEpochEnd(epoch, avgLoss);
			}
		}
	}
}