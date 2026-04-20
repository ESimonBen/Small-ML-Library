// gradientFn.inl

namespace MLCore::AutoGrad {
	template <typename T>
	GradFn<T>::GradFn(std::shared_ptr<Impl> impl) 
		: inputs {std::move(impl)}
	{}

	template <typename T>
	GradFn<T>::GradFn(std::vector<std::shared_ptr<Impl>> gradInput)
		: inputs(std::move(gradInput))
	{}
}