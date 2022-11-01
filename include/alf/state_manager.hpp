#ifndef ALF_STATE_MANAGER_HPP
#define ALF_STATE_MANAGER_HPP

#include <string>
#include <memory>
#include <vector>

#include <sqlite3.h>
#include <mlpack.hpp>


namespace alf {
    class State_manager {
    public:
		/**
		 * @brief Construct a new State_manager object
		 * @param path Path to SQLite file.
		 */
        explicit State_manager(std::string & path);
        ~State_manager();
        void load_labeled();
		void load_unlabeled();
		void remove_unlabeled(const arma::uvec& indices);
		std::shared_ptr<arma::mat> get_labeled() const;
		std::shared_ptr<arma::mat> get_unlabeled() const;
		std::shared_ptr<arma::Row<size_t>> get_labels() const;
		int get_labels_count() const;
    private:
		void open_db();
		void close_db();
        std::string m_path;
		std::shared_ptr<arma::mat> m_labeled;
		std::shared_ptr<arma::mat> m_unlabeled;
		std::shared_ptr<arma::Row<size_t>> m_labels;
		std::vector<int> m_unlabeled_index_mapping; // maps index of sample in armadillo matrix to index in database so: vector[i] = index in db
		sqlite3 * m_db;
	};
}

#endif