#ifndef ALF_QUERY_HPP
#define ALF_QUERY_HPP

#include <memory>

#include <mlpack.hpp>

#include <alf/state_manager.hpp>
#include <alf/model.hpp>

constexpr int default_train_frequency = 10;

using namespace mlpack;

namespace alf {

	template<class STRATEGY, class MODEL>
	class Query {
	public:
		/**
		 * @brief Construct a new Query object
		 * @param query Query to be executed.
		 */
		explicit Query(std::shared_ptr<MODEL> model, STRATEGY & query, int train_frequency = default_train_frequency):
			m_query(query),
			m_train_frequency(train_frequency),
            m_model(model) {};
		void execute();
	private:
		STRATEGY m_query;
		int m_train_frequency;
		int m_cycle_after_train = 0;
		std::shared_ptr<MODEL> m_model;
		State_manager m_state;
	};
}

#endif // ALF_QUERY_HPP


