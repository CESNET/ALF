#include <alf/pool_manager.hpp>

namespace alf {
	Pool_manager::Pool_manager(std::string & path): m_path(path){}

	Pool_manager::~Pool_manager() = default;

	arma::mat Pool_manager::load() const {
			arma::mat pool;
			mlpack::data::DatasetInfo datasetInfo;
			mlpack::data::Load(m_path, pool, datasetInfo, true);
			return pool;

	}

	void Pool_manager::removeSample(arma::mat& matrix, const arma::uvec& indices) {
		matrix.shed_cols(indices);
	}

}