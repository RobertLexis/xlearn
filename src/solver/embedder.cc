#include <sstream>
#include <algorithm>

#include "src/solver/embedder.h"

namespace xLearn
{

row_embedding embed_single_row(const SparseRow *row,
                               Model *model,
                               real_t norm)
{
    /*********************************************************
    *  linear term and bias term                            *
    *********************************************************/
    real_t value = 0;

    real_t *w = model->GetParameter_w();
    index_t aux_size = model->GetAuxiliarySize();
    row_embedding result;

    for (SparseRow::const_iterator iter = row->begin();
         iter != row->end(); ++iter)
    {
        value = w[iter->feat_id * aux_size] * iter->feat_val;
        result[{-1, iter->field_id}] += value;
    }

    // not need bias

    /*********************************************************
    *  latent factor                                        *
    *********************************************************/
    index_t align0 = aux_size * model->get_aligned_k();
    index_t align1 = model->GetNumField() * align0;
    int align = kAlign * aux_size;
    w = model->GetParameter_v();
    for (SparseRow::const_iterator iter_i = row->begin();
         iter_i != row->end(); ++iter_i)
    {
        index_t j1 = iter_i->feat_id;
        index_t f1 = iter_i->field_id;
        real_t v1 = iter_i->feat_val;
        for (SparseRow::const_iterator iter_j = iter_i + 1;
             iter_j != row->end(); ++iter_j)
        {
            index_t j2 = iter_j->feat_id;
            index_t f2 = iter_j->field_id;
            real_t v2 = iter_j->feat_val;
            if (!model->is_white(f1, f2))
            {
                continue;
            }

            real_t *w1_base = w + j1 * align1 + f2 * align0;
            real_t *w2_base = w + j2 * align1 + f1 * align0;
            __m128 XMMv = _mm_set1_ps(v1 * v2 * norm);

            // 注意清零
            __m128 XMMt = _mm_setzero_ps();
            value = 0;

            for (index_t d = 0; d < align0; d += align)
            {
                __m128 XMMw1 = _mm_load_ps(w1_base + d);
                __m128 XMMw2 = _mm_load_ps(w2_base + d);
                XMMt = _mm_add_ps(XMMt,
                                  _mm_mul_ps(
                                      _mm_mul_ps(XMMw1, XMMw2), XMMv));
            }

            XMMt = _mm_hadd_ps(XMMt, XMMt);
            XMMt = _mm_hadd_ps(XMMt, XMMt);
            _mm_store_ss(&value, XMMt);

            auto fields = std::minmax(iter_i->field_id, iter_j->field_id);

            result[{fields.first, fields.second}] += value;
        }
    }
    return result;
}

void embed_thread(DMatrix *matrix,
                  Model *model,
                  index_t start_idx,
                  index_t end_idx,
                  std::vector<std::pair<row_embedding, real_t>> &out)
{

    CHECK_GE(end_idx, start_idx);

    for (size_t i = start_idx; i < end_idx; ++i)
    {
        SparseRow *row = matrix->row[i];
        auto result = embed_single_row(row, model, matrix->norm[i]);
        out[i] = {result, matrix->Y[i]};
    }
}

void Embedder::EmbedCore(DMatrix *matrix,
                         std::vector<std::pair<row_embedding, real_t>> &out)
{
    CHECK_NOTNULL(matrix);
    CHECK_NE(out.empty(), true);
    CHECK_EQ(out.size(), matrix->row_length);

    int threadNumber = pool_->ThreadNumber();

    index_t row_len = matrix->row_length;

    // Predict in multi-thread
    for (int i = 0; i < threadNumber; ++i)
    {
        size_t start_idx = getStart(row_len, threadNumber, i);
        size_t end_idx = getEnd(row_len, threadNumber, i);

        pool_->enqueue(std::bind(embed_thread,
                                 matrix,
                                 model_,
                                 start_idx,
                                 end_idx,
                                 out));
    }
    // Wait all of the threads finish their job
    pool_->Sync(threadNumber);
}

void Embedder::Embed()
{
    static std::vector<std::pair<row_embedding, real_t>> out;

    DMatrix *matrix = nullptr;
    reader_->Reset();

    for (;;)
    {
        index_t tmp = reader_->Samples(matrix);
        if (tmp == 0)
        {
            break;
        }
        if (tmp != out.size())
        {
            out.resize(tmp);
        }

        EmbedCore(matrix, out);
        SaveResult(out);
    }
}

void Embedder::SaveResult(std::vector<std::pair<row_embedding, real_t>> &out)
{
    std::ofstream o_file(out_file_, std::ofstream::app);
    std::string delim = " ";

    for (auto row_embedding_2_y : out)
    {
        std::stringstream embedding_instance;
        row_embedding single_row_embedding = row_embedding_2_y.first;
        real_t y = row_embedding_2_y.second;
        embedding_instance << y;

        for (auto fields_2_value : single_row_embedding)
        {
            std::pair<index_t, index_t> fields = fields_2_value.first;
            real_t value = fields_2_value.second;
            embedding_instance << delim << fields.first << "~" << fields.second << ":" << value;
        }
        embedding_instance << "\n";
        o_file << embedding_instance.str();
    }
}
} // namespace xLearn