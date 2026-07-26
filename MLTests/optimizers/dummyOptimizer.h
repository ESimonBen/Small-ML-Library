/// dummyOptimizer.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/optimizers/optimizer.h>

using namespace MLCore::NN;
using namespace MLCore::TensorCore;
using namespace MLCore::Optimizers;
using namespace MLCore::Serialization;

template <typename T>
class DummyOptimizer : public Optimizer<T> {
public:
	DummyOptimizer(std::vector<std::reference_wrapper<Parameter<T>>> params, T learningRate, T weightDecay = static_cast<T>(0))
		: Optimizer<T>(params, learningRate, weightDecay)
	{}

	DummyOptimizer(std::vector<ParameterGroup<T>> groups)
		: Optimizer<T>(groups)
	{}

	virtual void Step() override {
		/// Simply clips gradients
		this->ClipGradients();
	}

	virtual std::string TypeName() const {
		return "DummyOptimizer";
	}

	virtual void SaveState(BinaryWriter& writer, const Module<T>& model) const override {
		/// No implementation needed
	}

	virtual void LoadState(BinaryReader& reader, Module<T>& model) override {
		/// No implementation needed
	}
};