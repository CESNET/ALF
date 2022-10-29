#pragma once 

#include <string>

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
        arma::mat load() const;
		static void removeSample(arma::mat &matrix, const arma::uvec& indices);
    private:
        std::string m_path;
    };
}