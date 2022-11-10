#include <alf/state_manager.hpp>

namespace alf {
	State_manager::State_manager(std::string & path): m_path(path), m_db(nullptr){}

	State_manager::~State_manager() = default;

	void State_manager::load_labeled() {
		open_db();
		sqlite3_stmt * stmt = nullptr;
		std::string sql = "SELECT * FROM labeled";
		int db_return_code = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
		if (db_return_code != SQLITE_OK) {
			mlpack::Log::Fatal << "Failed to fetch data: " << sqlite3_errmsg(m_db) << std::endl;
			close_db();
			throw std::runtime_error("Failed to fetch data.");
		}
		db_return_code = sqlite3_step(stmt);
		if (db_return_code != SQLITE_ROW) {
			close_db();
			throw std::runtime_error("No labeled data in database.");
		}
		int cols = sqlite3_column_count(stmt);
		int rows = 0;
		while (db_return_code == SQLITE_ROW) {
			rows++;
			db_return_code = sqlite3_step(stmt);
		}
        sqlite3_finalize(stmt);
        m_labeled = std::make_shared<arma::mat>(cols - 2, rows);
		m_labels = std::make_shared<arma::Row<size_t>>(rows);
		sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
		sqlite3_step(stmt);
		for (int row = 0; row < rows; ++row) {
			for (int col = 1; col < cols; col++) {
                if (col == cols - 1) {
					m_labels->at(row) = sqlite3_column_int(stmt, col);
				} else {
					m_labeled->at(col - 1, row) = sqlite3_column_double(stmt, col);
				}
			}
			row++;
			sqlite3_step(stmt);
		}
		sqlite3_finalize(stmt);
		close_db();
	}

	void State_manager::load_unlabeled() {
		open_db();
		sqlite3_stmt * stmt = nullptr;
		std::string sql = "SELECT * FROM unlabeled WHERE ANNOTATE = 0";
		int db_return_code = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
		if (db_return_code != SQLITE_OK) {
			mlpack::Log::Fatal << "Failed to fetch data: " << sqlite3_errmsg(m_db) << std::endl;
			close_db();
			throw std::runtime_error("Failed to fetch data.");
		}
		db_return_code = sqlite3_step(stmt);
		if (db_return_code != SQLITE_ROW) {
			close_db();
			throw std::runtime_error("No labeled data in database.");
		}
		int cols = sqlite3_column_count(stmt);
		int rows = 0;
		#pragma unroll
		while (db_return_code == SQLITE_ROW) {
			rows++;
			db_return_code = sqlite3_step(stmt);
		}
        sqlite3_finalize(stmt);

        m_unlabeled = std::make_shared<arma::mat>(cols - 2, rows);
		m_unlabeled_index_mapping = std::vector<int>(rows);
        sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
		sqlite3_step(stmt);
        for(int row = 0; row < rows; row++){
            for (int col = 0; col < cols - 1; col++) {
                if (col == 0) {
                    m_unlabeled_index_mapping[row] = sqlite3_column_int(stmt, col);
                } else {
                    m_unlabeled->at(col - 1, row) = sqlite3_column_double(stmt, col);
                }
            }
            sqlite3_step(stmt);
        }
		sqlite3_finalize(stmt);
		close_db();
	}


	void State_manager::annotate_unlabeled(const arma::uvec& indices) {
		open_db();
		sqlite3_stmt * stmt;
		std::string sql = "UPDATE unlabeled SET annotate = 1 WHERE id = ?";
		int rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
		if (rc != SQLITE_OK) {
			mlpack::Log::Fatal << "Failed to update data: " << sqlite3_errmsg(m_db) << std::endl;
			close_db();
			throw std::runtime_error("Failed to update data.");
		}
		for (int i = 0; i < indices.n_elem; i++) {
			sqlite3_bind_int(stmt, 1, m_unlabeled_index_mapping[indices[i]]);
			rc = sqlite3_step(stmt);
			if (rc != SQLITE_DONE) {
				mlpack::Log::Fatal << "Failed to update data: " << sqlite3_errmsg(m_db) << std::endl;
                sqlite3_finalize(stmt);
                close_db();
				throw std::runtime_error("Failed to update data.");
			}
			sqlite3_reset(stmt);
		}
        sqlite3_finalize(stmt);
        close_db();
    }


	void State_manager::open_db() {
		auto rc = sqlite3_open_v2(m_path.c_str(), &m_db, SQLITE_OPEN_READWRITE, nullptr);
		if (rc) {
			sqlite3_close_v2(m_db);
			mlpack::Log::Fatal << "Can't open database: " << sqlite3_errmsg(m_db) << std::endl;
			throw std::runtime_error("Can't open database. Wrong path?");
		} else {
			mlpack::Log::Info << "Opened database successfully" << std::endl;
		}
	}

	void State_manager::close_db() {
        sqlite3_close_v2(m_db);
        m_db = nullptr;
	}

	std::shared_ptr<arma::mat> State_manager::get_labeled() const {
		return m_labeled;
	}

	std::shared_ptr<arma::mat> State_manager::get_unlabeled() const {
		return m_unlabeled;
	}

	std::shared_ptr<arma::Row<size_t>> State_manager::get_labels() const {
		return m_labels;
	}

	int State_manager::get_labels_count() const {
		return static_cast<int>(m_labels->max() + 1);
	}

}