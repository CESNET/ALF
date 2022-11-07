#include <alf/query.hpp>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnreachableCode"
namespace alf {

	template<class STRATEGY, class MODEL>
	void Query<STRATEGY,MODEL>::execute() {
		m_state.load_unlabeled();
		if (m_train_frequency < m_cycle_after_train) {
			m_state.load_labeled();
			m_model->Train(*m_state.get_labeled(), *m_state.get_labels(), m_state.get_labels_count());
            m_cycle_after_train = 0;
		}
		arma::uvec flow_indices = m_query(m_model, m_state.get_labeled(), m_state.get_unlabeled());
		m_state.annotate_unlabeled(flow_indices);
		++m_cycle_after_train;
	}
}
#pragma clang diagnostic pop