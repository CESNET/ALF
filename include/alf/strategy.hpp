#ifndef ALF_STRATEGY_HPP
#define ALF_STRATEGY_HPP

#include <memory>
#include <vector>

#include <mlpack.hpp>

#include <alf/model.hpp>
#include <alf/util.hpp>

using namespace mlpack;

namespace alf {
    template<class MODEL>
    class StrategyBase {
    public:
        StrategyBase() = default;
        ~StrategyBase() = default;
        virtual arma::uvec select(
                std::shared_ptr<MODEL> model,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) = 0;
    };
    template<class MODEL>
    class RandomStrategy: public StrategyBase<MODEL> {
    public:
        explicit RandomStrategy(int count = 1): m_count(count) {};
        arma::uvec select(
                std::shared_ptr<MODEL> model,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) override;
    private:
        int m_count;
    };
    template<class MODEL>
    class UncertaintyLCStrategy: public StrategyBase<MODEL> {
    public:
        explicit UncertaintyLCStrategy(int count = 1, double threshold = 0): m_count(count), m_threshold(threshold) {};
        arma::uvec select(
                std::shared_ptr<MODEL> model,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) override;
	private:
        int m_count;
        double m_threshold;
    };
    template<class MODEL>
    class UncertaintyEntropyStrategy: public StrategyBase<MODEL> {
    public:
        explicit UncertaintyEntropyStrategy(int count = 1, double threshold = 0): m_count(count), m_threshold(threshold) {};
        arma::uvec select(
                std::shared_ptr<MODEL> model,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) override;
    private:
        int m_count;
        double m_threshold;
    };

	template<class MODEL>
	class QBDStrategy: public StrategyBase<MODEL> {
	public:
		explicit QBDStrategy(int count = 1, double threshold = 0): m_count(count), m_threshold(threshold) {};
		arma::uvec select(
				std::shared_ptr<MODEL> model,
				std::shared_ptr<arma::mat> labeled,
				std::shared_ptr<arma::mat> unlabeled) override;
	private:
		int m_count;
		double m_threshold;
	};

	template<class MODEL>
	class QBCEntropyStrategy: public StrategyBase<MODEL> {
	public:
		explicit QBCEntropyStrategy(int count = 1, double threshold = 0): m_count(count), m_threshold(threshold) {};
		arma::uvec select(
				std::shared_ptr<MODEL> model,
				std::shared_ptr<arma::mat> labeled,
				std::shared_ptr<arma::mat> unlabeled) override;
	private:
		int m_count;
		double m_threshold;
	};

	/**
	 * @brief Randomized Drusilla selection. Selects the farthest unlabeled sample from random selected labeled sample.
	 * Based on "Fast Approximate Furthest Neighbors with Data-Dependent Candidate Selection"
	 * @tparam MODEL
	 * @param model
	 * @param labeled
	 * @param unlabeled
	 * @return Vector of matrix indices.
	 */
	template<class MODEL>
	class FarthestSampleStrategy: public StrategyBase<MODEL> {
	public:
		explicit FarthestSampleStrategy(int count = 1, int l = 8, int m = 3): m_count(count), m_l(l), m_m(m)  {};
		arma::uvec select(
				std::shared_ptr<MODEL> model,
				std::shared_ptr<arma::mat> labeled,
				std::shared_ptr<arma::mat> unlabeled) override;
	private:
		int m_count;
		int m_l;
		int m_m;
	};

}

#endif //ALF_STRATEGY_HPP
