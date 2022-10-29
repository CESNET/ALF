#ifndef ALF_QUERY_HPP
#define ALF_QUERY_HPP

#include <mlpack.hpp>

using namespace mlpack;

namespace alf {
	template<class STRATEGY, class MODEL>
	class Query {
	public:
		/**
		 * @brief Construct a new Query object
		 * @param query Query to be executed.
		 */
		explicit Query(STRATEGY & query, MODEL & model);
		~Query();
		void execute();
	private:
		STRATEGY m_query;
		MODEL m_model;
		int m_train_frequency;
		int m_cycle_after_train;

	};
}

#endif // ALF_QUERY_HPP
