#ifndef ALF_POOL_MANAGER_HPP
#define ALF_POOL_MANAGER_HPP

#include <string>
#include <memory>

#include <mlpack.hpp>

namespace alf {
    class Pool_manager {
    public:
		/**
		 * @brief Construct a new Pool_manager object
		 * @param path Path to pool file.
		 */
        explicit Pool_manager(std::string & path);
        ~Pool_manager();
        void load();
		void removeSample(const arma::uvec& indices);
		void write();
		std::shared_ptr<arma::mat> getMatrix() const;
		std::shared_ptr<mlpack::data::DatasetInfo> getDatasetInfo() const;
    private:
        std::string m_path;
		std::shared_ptr<arma::mat> m_matrix;
		std::shared_ptr<mlpack::data::DatasetInfo> m_datasetInfo;
	};
}

#endif