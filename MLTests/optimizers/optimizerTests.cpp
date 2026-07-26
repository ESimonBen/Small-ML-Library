/// optimizerTests.cpp
#include <vector>
#include <doctest/doctest.h>
#include <mlCore/optimizers/optimizers.h>
#include <mlCore/operations/operations.h>

#include "dummyOptimizer.h"

using namespace MLCore::NN;
using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Optimizers;
using namespace MLCore::Operations;

TEST_SUITE("Optimizer Tests") {
	TEST_CASE("Base Optimizer Tests") {
		SUBCASE("Constructor initializes parameter groups") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(4.0f);

			Parameter<float> p{ A };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ std::ref(p) };

			DummyOptimizer opt(params, 0.1f, 0.01f);
			
			auto& groups = opt.ParamGroups();

			for (auto& group : groups) {
				CHECK(group.learningRate == doctest::Approx(0.1f));
				CHECK(group.weightDecay == doctest::Approx(0.01f));

				for (auto& ref : group.params) {
					auto& param = ref.get();
					auto& tensor = param.Data();
					size_t size = tensor.NumElements();

					for (size_t i = 0; i < size; ++i) {
						CHECK(tensor[i] == A[i]);
					}
				}
			}
		}

		SUBCASE("ZeroGrad zeros out parameter gradients") {
			ArenaAllocator allocator;

			Tensor<float> weight{ {1}, allocator };
			weight.Fill(0.0f);
			weight.SetRequiresGrad(true);

			Tensor<float> input{ {1}, allocator };
			input.Fill(3.0f);
			
			Tensor<float> target{ {1}, allocator };
			target.Fill(5.0f);

			Parameter<float> param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ std::ref(param) };

			DummyOptimizer opt{ params, 0.1f, .05f };

			auto predict = Multiply(input, weight, allocator);
			auto loss = MeanSquaredError(predict, target, Reduction::Mean, allocator);
			loss.Backward();

			auto weightGrad = weight.Grad();

			for (auto& val : weightGrad) {
				CHECK(val == -30.0f);
			}

			opt.ZeroGrad();

			for (auto& val : weightGrad) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("Gradient clipping correctly scales gradients") {
			ArenaAllocator allocator;

			Tensor<float> weight{ {1}, allocator };
			weight.Fill(1.0f);
			weight.SetRequiresGrad(true);

			Tensor<float> input{ {1}, allocator };
			input.Fill(3.0f);

			Tensor<float> target{ {1}, allocator };
			target.Fill(5.0f);

			Parameter<float> param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			DummyOptimizer opt{ params, 0.1f };

			auto predict = Multiply(input, weight, allocator);
			auto loss = MeanSquaredError(predict, target, Reduction::Mean, allocator);
			loss.Backward();

			auto weightGrad = weight.Grad();

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f);
			}

			opt.SetClipGradNorm(1.0f);
			opt.Step();

			for (auto& val : weightGrad) {
				CHECK(val == doctest::Approx(-1.0f));
			}
		}
	}

	TEST_CASE("SGD Tests") {
		SUBCASE("SGD takes in multiple parameters") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(4.0f);

			Tensor<float> B({ 2, 3 }, allocator);
			B.Fill(7.0f);

			Parameter paramA{ A };
			Parameter paramB{ B };

			std::vector<std::reference_wrapper<Parameter<float>>> params{ paramA, paramB };
			SGD opt{ params, 0.1f , 0.05f};

			auto& groups = opt.ParamGroups();

			for (auto& group : groups) {
				CHECK(group.learningRate == doctest::Approx(0.1f));
				CHECK(group.weightDecay == doctest::Approx(0.05f));

				size_t size = group.params.size();

				for (size_t i = 0; i < size - 1; ++i) {
					auto& param1 = group.params[i].get().Data();
					auto& param2 = group.params[i + 1].get().Data();

					size_t numElements = param1.NumElements(); /// Both tensors are the same size, so I can use this to iterate both

					for (size_t j = 0; j < numElements; ++j) {
						CHECK(param1[j] == A[j]);
						CHECK(param2[j] == B[j]);
					}
				}
			}
		}

		SUBCASE("SGD step correctly updates parameters") {
			ArenaAllocator allocator;

			Tensor<float> weight{ {1}, allocator };
			weight.Fill(1.0f);
			weight.SetRequiresGrad(true);

			Tensor<float> input{ {1}, allocator };
			input.Fill(3.0f);

			Tensor<float> target{ {1}, allocator };
			target.Fill(5.0f);

			Parameter param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			SGD opt{ params, 0.1f };

			auto predict = Multiply(input, weight, allocator);
			auto loss = MeanSquaredError(predict, target, Reduction::Mean, allocator);
			loss.Backward();

			auto weightGrad = weight.Grad();

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f);
			}

			opt.Step();

			for (auto& val : weight) {
				CHECK(val == 2.2f);
			}
		}

		SUBCASE("SGD weight decay correctly updates weight parameter") {
			ArenaAllocator allocator;

			Tensor<float> weight{ {1}, allocator };
			weight.Fill(1.0f);
			weight.SetRequiresGrad(true);

			Tensor<float> input{ {1}, allocator };
			input.Fill(3.0f);

			Tensor<float> target{ {1}, allocator };
			target.Fill(5.0f);

			Parameter<float> param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			SGD opt{ params, 0.1f, 0.05f };

			auto predict = Multiply(input, weight, allocator);
			auto loss = MeanSquaredError(predict, target, Reduction::Mean, allocator);
			loss.Backward();

			auto weightGrad = weight.Grad();

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f);
			}

			opt.Step();

			for (auto& val : weight) {
				CHECK(val == doctest::Approx(2.195f));
			}
		}

		SUBCASE("SGD correctly modifies multiple parameters") {
			ArenaAllocator allocator;

			/// Param A
			Tensor<float> weightA{ {1}, allocator };
			weightA.Fill(1.0f);
			weightA.SetRequiresGrad(true);

			Tensor<float> inputA{ {1}, allocator };
			inputA.Fill(3.0f);

			Tensor<float> targetA{ {1}, allocator };
			targetA.Fill(5.0f);

			Parameter<float> paramA{ weightA };

			/// Param B
			Tensor<float> weightB{ {1}, allocator };
			weightB.Fill(2.0f);
			weightB.SetRequiresGrad(true);

			Tensor<float> inputB{ {1}, allocator };
			inputB.Fill(4.0f);

			Tensor<float> targetB{ {1}, allocator };
			targetB.Fill(1.0f);

			Parameter<float> paramB{ weightB };

			std::vector<std::reference_wrapper<Parameter<float>>> params{ paramA, paramB };

			SGD opt{ params, 0.1f };

			/// Find gradient of Param A
			auto predictA = Multiply(inputA, weightA, allocator);
			auto lossA = MeanSquaredError(predictA, targetA, Reduction::Mean, allocator);
			lossA.Backward();

			/// Find gradient of Param B
			auto predictB = Multiply(inputB, weightB, allocator);
			auto lossB = MeanSquaredError(predictB, targetB, Reduction::Mean, allocator);
			lossB.Backward();

			auto weightAGrad = weightA.Grad();
			auto weightBGrad = weightB.Grad();

			for (auto& val : weightAGrad) {
				CHECK(val == -12.0f);
			}

			for (auto& val : weightBGrad) {
				CHECK(val == 56.0f);
			}

			opt.Step();

			for (auto& val : weightA) {
				CHECK(val == 2.2f);
			}

			for (auto& val : weightB) {
				CHECK(val == -3.6f);
			}
		}
	}

	TEST_CASE("SGDMomentum Tests") {
		SUBCASE("SGDMomentum takes in multiple parameters") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(4.0f);

			Tensor<float> B({ 2, 3 }, allocator);
			B.Fill(7.0f);

			Parameter paramA{ A };
			Parameter paramB{ B };

			std::vector<std::reference_wrapper<Parameter<float>>> params{ paramA, paramB };
			SGDMomentum opt{ params, 0.1f , 0.5f };

			auto& groups = opt.ParamGroups();

			for (auto& group : groups) {
				CHECK(group.learningRate == doctest::Approx(0.1f));
				CHECK(group.weightDecay == doctest::Approx(0.0f));

				size_t size = group.params.size();

				for (size_t i = 0; i < size - 1; ++i) {
					auto& param1 = group.params[i].get().Data();
					auto& param2 = group.params[i + 1].get().Data();

					size_t numElements = param1.NumElements(); /// Both tensors are the same size, so I can use this to iterate both

					for (size_t j = 0; j < numElements; ++j) {
						CHECK(param1[j] == A[j]);
						CHECK(param2[j] == B[j]);
					}
				}
			}
		}

		SUBCASE("SGDMomentum step correctly updates parameters with velocity") {
			ArenaAllocator allocator;

			Tensor<float> weight{ {1}, allocator };
			weight.Fill(1.0f);
			weight.SetRequiresGrad(true);

			Tensor<float> input{ {1}, allocator };
			input.Fill(3.0f);

			Tensor<float> target{ {1}, allocator };
			target.Fill(5.0f);

			Parameter param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			SGDMomentum opt{ params, 0.1f, 0.5f };

			auto predict = Multiply(input, weight, allocator);
			auto loss = MeanSquaredError(predict, target, Reduction::Mean, allocator);
			loss.Backward();

			auto weightGrad = weight.Grad();

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f);
			}

			opt.Step();

			for (auto& val : weight) {
				CHECK(val == 2.2f);
			}

			opt.Step(); /// To test the velocity

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f); /// Same as before
			}

			for (auto& val : weight) {
				CHECK(val == 4.0f);
			}
		}

		SUBCASE("SGDMomentum weight decay correctly updates weight parameter") {
			ArenaAllocator allocator;

			Tensor<float> weight{ {1}, allocator };
			weight.Fill(1.0f);
			weight.SetRequiresGrad(true);

			Tensor<float> input{ {1}, allocator };
			input.Fill(3.0f);

			Tensor<float> target{ {1}, allocator };
			target.Fill(5.0f);

			Parameter<float> param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			SGDMomentum opt{ params, 0.1f, 0.5f, 0.05f };

			auto predict = Multiply(input, weight, allocator);
			auto loss = MeanSquaredError(predict, target, Reduction::Mean, allocator);
			loss.Backward();

			auto weightGrad = weight.Grad();

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f);
			}

			opt.Step();

			for (auto& val : weight) {
				CHECK(val == doctest::Approx(2.195f));
			}
		}

		SUBCASE("SGDMomentum correctly modifies multiple parameters") {
			ArenaAllocator allocator;

			/// Param A
			Tensor<float> weightA{ {1}, allocator };
			weightA.Fill(1.0f);
			weightA.SetRequiresGrad(true);

			Tensor<float> inputA{ {1}, allocator };
			inputA.Fill(3.0f);

			Tensor<float> targetA{ {1}, allocator };
			targetA.Fill(5.0f);

			Parameter<float> paramA{ weightA };

			/// Param B
			Tensor<float> weightB{ {1}, allocator };
			weightB.Fill(2.0f);
			weightB.SetRequiresGrad(true);

			Tensor<float> inputB{ {1}, allocator };
			inputB.Fill(4.0f);

			Tensor<float> targetB{ {1}, allocator };
			targetB.Fill(1.0f);

			Parameter<float> paramB{ weightB };

			std::vector<std::reference_wrapper<Parameter<float>>> params{ paramA, paramB };

			SGDMomentum opt{ params, 0.1f, 0.5f };

			/// Find gradient of Param A
			auto predictA = Multiply(inputA, weightA, allocator);
			auto lossA = MeanSquaredError(predictA, targetA, Reduction::Mean, allocator);
			lossA.Backward();

			/// Find gradient of Param B
			auto predictB = Multiply(inputB, weightB, allocator);
			auto lossB = MeanSquaredError(predictB, targetB, Reduction::Mean, allocator);
			lossB.Backward();

			auto weightAGrad = weightA.Grad();
			auto weightBGrad = weightB.Grad();

			for (auto& val : weightAGrad) {
				CHECK(val == -12.0f);
			}

			for (auto& val : weightBGrad) {
				CHECK(val == 56.0f);
			}

			opt.Step();

			for (auto& val : weightA) {
				CHECK(val == 2.2f);
			}

			for (auto& val : weightB) {
				CHECK(val == -3.6f);
			}
		}
	}

	TEST_CASE("Adam Tests") {
		SUBCASE("Adam takes in multiple parameters") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(4.0f);

			Tensor<float> B({ 2, 3 }, allocator);
			B.Fill(7.0f);

			Parameter paramA{ A };
			Parameter paramB{ B };

			std::vector<std::reference_wrapper<Parameter<float>>> params{ paramA, paramB };
			Adam opt{ params, 0.1f };

			auto& groups = opt.ParamGroups();

			for (auto& group : groups) {
				CHECK(group.learningRate == doctest::Approx(0.1f));
				CHECK(group.weightDecay == doctest::Approx(0.0f));

				size_t size = group.params.size();

				for (size_t i = 0; i < size - 1; ++i) {
					auto& param1 = group.params[i].get().Data();
					auto& param2 = group.params[i + 1].get().Data();

					size_t numElements = param1.NumElements(); /// Both tensors are the same size, so I can use this to iterate both

					for (size_t j = 0; j < numElements; ++j) {
						CHECK(param1[j] == A[j]);
						CHECK(param2[j] == B[j]);
					}
				}
			}
		}

		SUBCASE("Adam step correctly updates parameters with 1st and 2nd moments") {
			ArenaAllocator allocator;

			Tensor<float> weight{ {1}, allocator };
			weight.Fill(1.0f);
			weight.SetRequiresGrad(true);

			Tensor<float> input{ {1}, allocator };
			input.Fill(3.0f);

			Tensor<float> target{ {1}, allocator };
			target.Fill(5.0f);

			Parameter param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			Adam opt{ params, 0.1f };

			auto predict = Multiply(input, weight, allocator);
			auto loss = MeanSquaredError(predict, target, Reduction::Mean, allocator);
			loss.Backward();

			auto weightGrad = weight.Grad();

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f);
			}

			opt.Step();

			for (auto& val : weight) {
				CHECK(val == 1.1f);
			}

			opt.Step(); /// To test the velocity

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f); /// Same as before
			}

			for (auto& val : weight) {
				CHECK(val == doctest::Approx(1.2f));
			}
		}

		SUBCASE("Adam weight decay correctly updates weight parameter") {
			ArenaAllocator allocator;

			Tensor<float> weight{ {1}, allocator };
			weight.Fill(1.0f);
			weight.SetRequiresGrad(true);

			Tensor<float> input{ {1}, allocator };
			input.Fill(3.0f);

			Tensor<float> target{ {1}, allocator };
			target.Fill(5.0f);

			Parameter<float> param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			Adam opt{ params, 0.1f, 0.05f };

			auto predict = Multiply(input, weight, allocator);
			auto loss = MeanSquaredError(predict, target, Reduction::Mean, allocator);
			loss.Backward();

			auto weightGrad = weight.Grad();

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f);
			}

			opt.Step();

			for (auto& val : weight) {
				CHECK(val == doctest::Approx(1.10f));
			}
		}

		SUBCASE("Adam correctly modifies multiple parameters") {
			ArenaAllocator allocator;

			/// Param A
			Tensor<float> weightA{ {1}, allocator };
			weightA.Fill(1.0f);
			weightA.SetRequiresGrad(true);

			Tensor<float> inputA{ {1}, allocator };
			inputA.Fill(3.0f);

			Tensor<float> targetA{ {1}, allocator };
			targetA.Fill(5.0f);

			Parameter<float> paramA{ weightA };

			/// Param B
			Tensor<float> weightB{ {1}, allocator };
			weightB.Fill(2.0f);
			weightB.SetRequiresGrad(true);

			Tensor<float> inputB{ {1}, allocator };
			inputB.Fill(4.0f);

			Tensor<float> targetB{ {1}, allocator };
			targetB.Fill(1.0f);

			Parameter<float> paramB{ weightB };

			std::vector<std::reference_wrapper<Parameter<float>>> params{ paramA, paramB };

			Adam opt{ params, 0.1f, 0.5f };

			/// Find gradient of Param A
			auto predictA = Multiply(inputA, weightA, allocator);
			auto lossA = MeanSquaredError(predictA, targetA, Reduction::Mean, allocator);
			lossA.Backward();

			/// Find gradient of Param B
			auto predictB = Multiply(inputB, weightB, allocator);
			auto lossB = MeanSquaredError(predictB, targetB, Reduction::Mean, allocator);
			lossB.Backward();

			auto weightAGrad = weightA.Grad();
			auto weightBGrad = weightB.Grad();

			for (auto& val : weightAGrad) {
				CHECK(val == -12.0f);
			}

			for (auto& val : weightBGrad) {
				CHECK(val == 56.0f);
			}

			opt.Step();

			for (auto& val : weightA) {
				CHECK(val == 1.1f);
			}

			for (auto& val : weightB) {
				CHECK(val == 1.9f);
			}
		}
	}

	TEST_CASE("AdamW Tests") {
		SUBCASE("AdamW takes in multiple parameters") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(4.0f);

			Tensor<float> B({ 2, 3 }, allocator);
			B.Fill(7.0f);

			Parameter paramA{ A };
			Parameter paramB{ B };

			std::vector<std::reference_wrapper<Parameter<float>>> params{ paramA, paramB };
			AdamW opt{ params, 0.1f };

			auto& groups = opt.ParamGroups();

			for (auto& group : groups) {
				CHECK(group.learningRate == doctest::Approx(0.1f));
				CHECK(group.weightDecay == doctest::Approx(0.0f));

				size_t size = group.params.size();

				for (size_t i = 0; i < size - 1; ++i) {
					auto& param1 = group.params[i].get().Data();
					auto& param2 = group.params[i + 1].get().Data();

					size_t numElements = param1.NumElements(); /// Both tensors are the same size, so I can use this to iterate both

					for (size_t j = 0; j < numElements; ++j) {
						CHECK(param1[j] == A[j]);
						CHECK(param2[j] == B[j]);
					}
				}
			}
		}

		SUBCASE("AdamW step correctly updates parameters with 1st and 2nd moments") {
			ArenaAllocator allocator;

			Tensor<float> weight{ {1}, allocator };
			weight.Fill(1.0f);
			weight.SetRequiresGrad(true);

			Tensor<float> input{ {1}, allocator };
			input.Fill(3.0f);

			Tensor<float> target{ {1}, allocator };
			target.Fill(5.0f);

			Parameter param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			AdamW opt{ params, 0.1f };

			auto predict = Multiply(input, weight, allocator);
			auto loss = MeanSquaredError(predict, target, Reduction::Mean, allocator);
			loss.Backward();

			auto weightGrad = weight.Grad();

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f);
			}

			opt.Step();

			for (auto& val : weight) {
				CHECK(val == 1.1f);
			}

			opt.Step(); /// To test the velocity

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f); /// Same as before
			}

			for (auto& val : weight) {
				CHECK(val == doctest::Approx(1.2f));
			}
		}

		SUBCASE("AdamW weight decay correctly updates weight parameter") {
			ArenaAllocator allocator;

			Tensor<float> weight{ {1}, allocator };
			weight.Fill(1.0f);
			weight.SetRequiresGrad(true);

			Tensor<float> input{ {1}, allocator };
			input.Fill(3.0f);

			Tensor<float> target{ {1}, allocator };
			target.Fill(5.0f);

			Parameter<float> param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			AdamW opt{ params, 0.1f, 0.05f };

			auto predict = Multiply(input, weight, allocator);
			auto loss = MeanSquaredError(predict, target, Reduction::Mean, allocator);
			loss.Backward();

			auto weightGrad = weight.Grad();

			for (auto& val : weightGrad) {
				CHECK(val == -12.0f);
			}

			opt.Step();

			for (auto& val : weight) {
				CHECK(val == doctest::Approx(1.095f));
			}
		}

		SUBCASE("AdamW correctly modifies multiple parameters") {
			ArenaAllocator allocator;

			/// Param A
			Tensor<float> weightA{ {1}, allocator };
			weightA.Fill(1.0f);
			weightA.SetRequiresGrad(true);

			Tensor<float> inputA{ {1}, allocator };
			inputA.Fill(3.0f);

			Tensor<float> targetA{ {1}, allocator };
			targetA.Fill(5.0f);

			Parameter<float> paramA{ weightA };

			/// Param B
			Tensor<float> weightB{ {1}, allocator };
			weightB.Fill(2.0f);
			weightB.SetRequiresGrad(true);

			Tensor<float> inputB{ {1}, allocator };
			inputB.Fill(4.0f);

			Tensor<float> targetB{ {1}, allocator };
			targetB.Fill(1.0f);

			Parameter<float> paramB{ weightB };

			std::vector<std::reference_wrapper<Parameter<float>>> params{ paramA, paramB };

			AdamW opt{ params, 0.1f, 0.5f };

			/// Find gradient of Param A
			auto predictA = Multiply(inputA, weightA, allocator);
			auto lossA = MeanSquaredError(predictA, targetA, Reduction::Mean, allocator);
			lossA.Backward();

			/// Find gradient of Param B
			auto predictB = Multiply(inputB, weightB, allocator);
			auto lossB = MeanSquaredError(predictB, targetB, Reduction::Mean, allocator);
			lossB.Backward();

			auto weightAGrad = weightA.Grad();
			auto weightBGrad = weightB.Grad();

			for (auto& val : weightAGrad) {
				CHECK(val == -12.0f);
			}

			for (auto& val : weightBGrad) {
				CHECK(val == 56.0f);
			}

			opt.Step();

			for (auto& val : weightA) {
				CHECK(val == doctest::Approx(1.05f));
			}

			for (auto& val : weightB) {
				CHECK(val == 1.8f);
			}
		}
	}
}