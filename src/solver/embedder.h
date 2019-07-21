#ifndef XLEARN_SOLVER_EMBEDDER_H_
#define XLEARN_SOLVER_EMBEDDER_H_
#include <string>

#include "src/base/thread_pool.h"
#include "src/data/model_parameters.h"
#include "src/reader/reader.h"

namespace xLearn
{
typedef std::map<std::pair<index_t, index_t>, real_t> row_embedding;

class Embedder
{
private:
    /* data */
public:
    Embedder(/* args */){};
    ~Embedder(){};

    void Initialize(Reader *reader,
                    Model *model,
                    const std::string &out_file,
                    ThreadPool *pool)
    {
        CHECK_NOTNULL(reader);
        CHECK_NOTNULL(model);
        CHECK_NOTNULL(pool);

        reader_ = reader;
        model_ = model;
        out_file_ = out_file;
        pool_ = pool;
    }

    void Embed();
    void EmbedCore(DMatrix *matrix,
                   std::vector<std::pair<row_embedding, real_t>> &out);
    void SaveResult(std::vector<std::pair<row_embedding, real_t>> &out);

protected:
    Reader *reader_;
    Model *model_;
    std::string out_file_;
    ThreadPool *pool_;
};

// Embedder::Embedder(/* args */)
// {
// }

// Embedder::~Embedder()
// {
// }

} // namespace xLearn
#endif // XLEARN_SOLVER_EMBEDDER_H_