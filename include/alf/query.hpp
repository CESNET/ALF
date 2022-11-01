#ifndef ALF_QUERY_HPP
#define ALF_QUERY_HPP

#include <memory>

#include <mlpack.hpp>

#include <alf/state_manager.hpp>

using namespace mlpack;

namespace alf {

	enum Phase {
		COLD,
		WARM,
		HOT,
		FEVER
	};

	template<class STRATEGY, class PHASE_POLICY>
	class Query {
	public:
		/**
		 * @brief Construct a new Query object
		 * @param query Query to be executed.
		 */
		explicit Query(STRATEGY & query, PHASE_POLICY & phase_policy, int train_frequency):
			m_query(query),
			m_train_frequency(train_frequency),
			m_cycle_after_train(0),
			m_phase(Phase::COLD),
			m_phase_policy(phase_policy) {
				m_rf = std::make_shared<RandomForest<>>();
			};
		void execute();
		void send_to_anotator(arma::mat & samples);
	private:
		STRATEGY m_query;
		PHASE_POLICY m_phase_policy;
		int m_train_frequency;
		int m_cycle_after_train;
		Phase m_phase;
		std::shared_ptr<RandomForest<>> m_rf;
		State_manager m_state;
	};
}

#endif // ALF_QUERY_HPP
