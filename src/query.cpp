#include <alf/query.hpp>

namespace alf {

	template<class STRATEGY>
	void Query<STRATEGY>::execute() {
		m_state.load_unlabeled();
		if (m_cycle_after_train > m_train_frequency) {
			m_state.load_labeled();
			m_rf->Train(*m_state.get_labeled(), *m_state.get_labels(), m_state.get_labels_count());
		}
		arma::uvec flow_indices = m_query(m_rf, m_state.get_labeled(), m_state.get_unlabeled());
		send_to_anotator(m_state.get_unlabeled()->cols(flow_indices));
		m_state.remove_unlabeled(flow_indices);
		++m_cycle_after_train;
	}

}