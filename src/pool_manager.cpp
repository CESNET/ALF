#include <alf/pool_manager.hpp>

namespace alf {
	Pool_manager::Pool_manager(std::string & path): m_path(path){}

	Pool_manager::~Pool_manager() = default;

	void Pool_manager::load() {
		arma::mat matrix;
		mlpack::data::DatasetInfo datasetInfo;
		mlpack::data::Load(m_path, matrix, datasetInfo, true);
		m_matrix = std::make_shared<arma::mat>(matrix);
		m_datasetInfo = std::make_shared<mlpack::data::DatasetInfo>(datasetInfo);
	}

	void Pool_manager::removeSample(const arma::uvec& indices) {
		m_matrix->shed_cols(indices);
	}

	void Pool_manager::write() {
		mlpack::data::Save(m_path, *m_matrix);
	}

	std::shared_ptr<arma::mat> Pool_manager::getMatrix() const {
		return m_matrix;
	}

	std::shared_ptr<mlpack::data::DatasetInfo> Pool_manager::getDatasetInfo() const {
		return m_datasetInfo;
	}


}