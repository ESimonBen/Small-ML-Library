// gradientFn.inl

namespace MLCore::AutoGrad {
	template <typename T>
	GradFn<T>::GradFn(Tensor* gradInput) {
		inputs.push_back(gradInput);
	}

	template <typename T>
	GradFn<T>::GradFn(const std::vector<Tensor*>& gradInput)
		: inputs(gradInput)
	{}
}