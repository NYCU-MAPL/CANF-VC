/**
 * COPYRIGHT 2019 ETH Zurich
 *
 * LINUX:

 - GPU: nvcc 9 and gcc 5.5 works
 - without: ???

 *
 * MACOS:
 *
 * CC=clang++ -std=libc++
 * MACOSX_DEPLOYMENT_TARGET=10.14
 *
 * BASED on
 *
 * https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
 */

#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <algorithm>
#include <string>
#include <chrono>
#include <numeric>
#include <iterator>

#include <bitset>

using cdf_t = uint16_t;

/** Class to read byte string bit by bit */
class InCacheString
{
private:
    uint8_t cache = 0;
    uint8_t cached_bits = 0; // num
    size_t in_ptr = 0;

public:
    InCacheString() = default;
    InCacheString(const std::string &in) : in_(in){};

    std::string in_;
    void get(uint32_t &value)
    {
        if (cached_bits == 0)
        {
            if (in_ptr == in_.size())
            {
                value <<= 1;
                return;
            }
            /// Read 1 byte
            cache = (uint8_t)in_[in_ptr];
            in_ptr++;
            cached_bits = 8;
        }
        value <<= 1;
        value |= (cache >> (cached_bits - 1)) & 1;
        cached_bits--;
    }
};

/** Class to save output bit by bit to a byte string */
class OutCacheString
{
private:
    std::string out = "";
    uint8_t cache = 0;
    uint8_t count = 0;

public:
    void append(const int bit)
    {
        cache <<= 1;
        cache |= bit;
        count += 1;
        if (count == 8)
        {
            out.append(reinterpret_cast<const char *>(&cache), 1);
            count = 0;
        }
    }
    void append_bit_and_pending(const int bit, uint64_t &pending_bits)
    {
        append(bit);
        while (pending_bits > 0)
        {
            append(!bit);
            pending_bits -= 1;
        }
    }
    void finish()
    {
        if (count > 0)
        {
            for (int i = count; i < 8; ++i)
            {
                append(0);
            }
            assert(count == 0);
        }
    }
    py::bytes flush()
    {
        auto ret = py::bytes(out);
        out = "";
        cache = 0;
        count = 0;
        return ret;
    }
};
const int precision = 16;

class rans_state
{
public:
    uint32_t low;
    uint32_t high;
    uint32_t value;
    uint64_t pending_bits;
    rans_state()
    {
        clear();
    }
    void clear()
    {
        low = 0;
        high = 0xFFFFFFFFU;
        value = 0;
        pending_bits = 0;
    }

    void initialize(InCacheString &in_cache)
    {
        for (int i = 0; i < 32; ++i)
        {
            in_cache.get(value);
        }
    }
    void finish(OutCacheString &out_cache)
    {
        pending_bits += 1;

        if (pending_bits)
        {
            if (low < 0x40000000U)
            {
                out_cache.append_bit_and_pending(0, pending_bits);
            }
            else
            {
                out_cache.append_bit_and_pending(1, pending_bits);
            }
        }
    }
    void update_high_low(const uint32_t c_low, const uint32_t c_high)
    {
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low = (low) + ((span * static_cast<uint64_t>(c_low)) >> precision);
    }
    void single_encode(
        const uint32_t c_low,
        const uint32_t c_high,
        OutCacheString &out_cache)
    {

        update_high_low(c_low, c_high);

        while (true)
        {
            if (high < 0x80000000U)
            {
                out_cache.append_bit_and_pending(0, pending_bits);
                low <<= 1;
                high <<= 1;
                high |= 1;
            }
            else if (low >= 0x80000000U)
            {
                out_cache.append_bit_and_pending(1, pending_bits);
                low <<= 1;
                high <<= 1;
                high |= 1;
            }
            else if (low >= 0x40000000U && high < 0xC0000000U)
            {
                pending_bits++;
                low <<= 1;
                low &= 0x7FFFFFFF;
                high <<= 1;
                high |= 0x80000001;
            }
            else
            {
                break;
            }
        }
        return;
    }
    void single_decode(
        const uint32_t c_low,
        const uint32_t c_high,
        InCacheString &in_cache)
    {

        update_high_low(c_low, c_high);

        while (true)
        {
            if (low >= 0x80000000U || high < 0x80000000U)
            {
                low <<= 1;
                high <<= 1;
                high |= 1;
                in_cache.get(value);
            }
            else if (low >= 0x40000000U && high < 0xC0000000U)
            {
                /**
             * 0100 0000 ... <= value <  1100 0000 ...
             * <=>
             * 0100 0000 ... <= value <= 1011 1111 ...
             * <=>
             * value starts with 01 or 10.
             * 01 - 01 == 00  |  10 - 01 == 01
             * i.e., with shifts
             * 01A -> 0A  or  10A -> 1A, i.e., discard 2SB as it's all the same while we are in
             *    near convergence
             */
                low <<= 1;
                low &= 0x7FFFFFFFU; // make MSB 0
                high <<= 1;
                high |= 0x80000001U; // add 1 at the end, retain MSB = 1
                value -= 0x40000000U;
                in_cache.get(value);
            }
            else
            {
                break;
            }
        }

        return;
    }
};

/// This is set by setup.py if CUDA support is desired
#ifdef COMPILE_CUDA
/// All these are defined in torchac_kernel.cu
cdf_t *malloc_cdf(const int N, const int Lp);
void free_cdf(cdf_t *cdf_mem);
void calculate_cdf(
    const at::Tensor &targets,
    const at::Tensor &means,
    const at::Tensor &log_scales,
    const at::Tensor &logit_probs_softmax,
    cdf_t *cdf_mem,
    const int K, const int Lp, const int N_cdf);

#endif // COMPILE_CUDA

// -----------------------------------------------------------------------------

/** Encode symbols `sym` with CDF represented by `cdf_ptr`. NOTE: this is not exposted to python. */
std::vector<py::bytes>
encode(
    rans_state &state,
    OutCacheString &out_cache,
    OutCacheString &outbound_cache,
    const at::Tensor &_cdf,
    const at::Tensor &cdf_length,
    const at::Tensor &indexes,
    const at::Tensor &sym)
{

#ifdef VERBOSE
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    TORCH_CHECK(!_cdf.is_cuda(), "cdf must be on CPU!");
    const auto s = _cdf.sizes();
    TORCH_CHECK(s.size() == 2, "Invalid size for cdf! Expected NLp");

    const int Lp = s[1];
    const auto cdf_acc = _cdf.accessor<int16_t, 2>();
    const cdf_t *cdf = (cdf_t *)cdf_acc.data();
    auto cdf_length_ = cdf_length.accessor<int16_t, 1>();

    const int N_sym = at::numel(indexes);
    const auto idx_reshaped = at::reshape(indexes, {N_sym});
    auto idx_ = idx_reshaped.accessor<int16_t, 1>();
    const auto sym_reshaped = at::reshape(sym, {N_sym});
    auto sym_ = sym_reshaped.accessor<int16_t, 1>();

    for (int i = 0; i < N_sym; ++i)
    {
        const int idx = idx_[i];
        const int max_symbol = cdf_length_[idx];

        int16_t sym_i = sym_[i];
        uint32_t overflow = 0;

        if (sym_i < 0)
        {
            overflow = -2 * sym_i - 1;
            sym_i = max_symbol;
        }
        else if (sym_i >= max_symbol)
        {
            overflow = 2 * (sym_i - max_symbol);
            sym_i = max_symbol;
        }

        const int offset = idx * Lp;
        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        state.single_encode(c_low, c_high, out_cache);

        // Variable length coding for out of bound symbols.
        if (sym_i == max_symbol)
        {
            int32_t width = 0;
            while (overflow >> width != 0)
            {
                ++width;
            }

            uint32_t val = width;
            while (val > 0)
            {
                outbound_cache.append(1);
                val--;
            }
            outbound_cache.append(0);

            for (int32_t j = 0; j < width; ++j)
            {
                val = (overflow >> j) & 1;
                outbound_cache.append(val);
            }
        }
    }

    state.finish(out_cache);
    out_cache.finish();
    outbound_cache.finish();
    state.clear();

#ifdef VERBOSE
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 << std::endl;
#endif

    return {out_cache.flush(), outbound_cache.flush()};
}

void encode_cdf_(
    rans_state &state,
    OutCacheString &out_cache,
    const at::Tensor &_cdf,
    const at::Tensor &sym)
{

#ifdef VERBOSE
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    TORCH_CHECK(!_cdf.is_cuda(), "cdf must be on CPU!");
    const auto s = _cdf.sizes();
    TORCH_CHECK(s.size() == 2, "Invalid size for cdf! Expected NLp");

    const int Lp = s[1];
    const auto cdf_acc = _cdf.accessor<int16_t, 2>();
    const cdf_t *cdf = (cdf_t *)cdf_acc.data();
    const int max_symbol = Lp - 2;

    const int N_sym = at::numel(sym);
    TORCH_CHECK(s[0] == N_sym, "Invalid size for cdf! Expected NLp");
    const auto sym_reshaped = at::reshape(sym, {N_sym});
    auto sym_ = sym_reshaped.accessor<int16_t, 1>();

    for (int i = 0; i < N_sym; ++i)
    {
        const int16_t sym_i = sym_[i];

        const int offset = i * Lp;
        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        state.single_encode(c_low, c_high, out_cache);
    }
}

/** See torchac.py */

class TorchEncoder
{
public:
    TorchEncoder() = default;
    void set_cdf(const at::Tensor &cdf)
    {
        _cdf = cdf.cpu();
    };
    void set_cdf_index(const at::Tensor &cdf, const at::Tensor &cdf_length)
    {
        _cdf = cdf.cpu();
        _cdf_length = cdf_length.cpu();
    };
    std::vector<py::bytes> encode_cdf_index(
        const at::Tensor &cdf, /* NLp */
        const at::Tensor &cdf_length,
        const at::Tensor &indexes,
        const at::Tensor &sym);
    std::vector<py::bytes> encode_index(
        const at::Tensor &indexes,
        const at::Tensor &sym);
    void encode_cdf(
        const at::Tensor &sym,
        const at::Tensor &cdf);
    std::vector<py::bytes> flush()
    {
        _state.finish(_out_cache);
        _out_cache.finish();
        _outbound_cache.finish();
        _state.clear();
        return {_out_cache.flush(), _outbound_cache.flush()};
    }

    at::Tensor _cdf;
    at::Tensor _cdf_length;
    rans_state _state;
    OutCacheString _out_cache;
    OutCacheString _outbound_cache;
};

std::vector<py::bytes>
TorchEncoder::encode_cdf_index(
    const at::Tensor &cdf, /* NLp */
    const at::Tensor &cdf_length,
    const at::Tensor &indexes,
    const at::Tensor &sym)
{
    rans_state state;
    OutCacheString out_cache;
    OutCacheString outbound_cache;
    return encode(state, out_cache, outbound_cache, cdf.cpu(), cdf_length.cpu(), indexes.cpu(), sym.cpu());
}

std::vector<py::bytes>
TorchEncoder::encode_index(
    const at::Tensor &indexes,
    const at::Tensor &sym)
{
    return encode(_state, _out_cache, _outbound_cache, _cdf, _cdf_length, indexes.cpu(), sym.cpu());
}

void TorchEncoder::encode_cdf(
    const at::Tensor &sym,
    const at::Tensor &cdf)
{
    encode_cdf_(_state, _out_cache, cdf.cpu(), sym.cpu());
}

//------------------------------------------------------------------------------

cdf_t binsearch(const cdf_t *cdf, cdf_t target, cdf_t max_sym, const int offset) /* i * Lp */
{
    cdf_t left = 0;
    cdf_t right = max_sym + 1; // len(cdf) == max_sym + 2

    while (left + 1 < right)
    {
        // left and right will be < 0x10000 in practice, so left+right fits in uint16_t...
        const auto m = static_cast<const cdf_t>((left + right) / 2);
        const auto v = cdf[offset + m];
        if (v < target)
        {
            left = m;
        }
        else if (v > target)
        {
            right = m;
        }
        else
        {
            return m;
        }
    }

    return left;
}

at::Tensor decode(
    rans_state &state,
    InCacheString &in_cache,
    InCacheString outbound_cache,
    const at::Tensor &_cdf,
    const at::Tensor &cdf_length,
    const at::Tensor &indexes)
{

#ifdef VERBOSE
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    TORCH_CHECK(!_cdf.is_cuda(), "cdf must be on CPU!");
    const auto s = _cdf.sizes();
    TORCH_CHECK(s.size() == 2, "Invalid size for cdf! Expected NLp");

    const int Lp = s[1];
    const auto cdf_acc = _cdf.accessor<int16_t, 2>();
    const cdf_t *cdf = (cdf_t *)cdf_acc.data();
    auto cdf_length_ = cdf_length.accessor<int16_t, 1>();

    const int N_sym = at::numel(indexes);
    const auto idx_reshaped = at::reshape(indexes, {N_sym});
    auto idx_ = idx_reshaped.accessor<int16_t, 1>();

    // 16 bit!
    auto out = torch::empty({N_sym}, at::kShort);
    auto out_ = out.accessor<int16_t, 1>();

    const uint32_t c_count = 0x10000U;

    for (int i = 0; i < N_sym; ++i)
    {
        const int idx = idx_[i];
        const int max_symbol = cdf_length_[idx];

        // TODO: remove cast
        const uint64_t span = static_cast<uint64_t>(state.high) - static_cast<uint64_t>(state.low) + 1;
        // always < 0x10000 ???
        const uint16_t count = ((static_cast<uint64_t>(state.value) - static_cast<uint64_t>(state.low) + 1) * c_count - 1) / span;

        const int offset = idx * Lp;
        int16_t sym_i = binsearch(cdf, count, (cdf_t)max_symbol, offset);

        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        // Variable length coding for out of bound symbols.
        if (sym_i == max_symbol)
        {
            int32_t width = 0;
            uint32_t val;

            do
            {
                val = 0;
                outbound_cache.get(val);
                width += (int32_t)val;
            } while (val == 1);

            uint32_t overflow = 0;

            for (int32_t j = 0; j < width; ++j)
            {
                val = 0;
                outbound_cache.get(val);

                overflow |= val << j;
            }

            if (overflow > 0)
            {
                sym_i = overflow >> 1;
                sym_i = overflow & 1 ? -sym_i - 1 : sym_i + max_symbol;
            }
        }

        out_[i] = (int16_t)sym_i;

        if (i == N_sym - 1)
        {
            break;
        }

        state.single_decode(c_low, c_high, in_cache);
    }

#ifdef VERBOSE
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 << std::endl;
#endif

    return out.reshape_as(indexes);
}

at::Tensor decode_cdf_(
    rans_state &state,
    InCacheString &in_cache,
    const at::Tensor &_cdf)
{

#ifdef VERBOSE
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    TORCH_CHECK(!_cdf.is_cuda(), "cdf must be on CPU!");
    const auto s = _cdf.sizes();
    TORCH_CHECK(s.size() == 2, "Invalid size for cdf! Expected NLp");

    const int Lp = s[1];
    const auto cdf_acc = _cdf.accessor<int16_t, 2>();
    const cdf_t *cdf = (cdf_t *)cdf_acc.data();
    const int max_symbol = Lp - 2;

    const int N_sym = s[0];

    // 16 bit!
    auto out = torch::empty({N_sym}, at::kShort);
    auto out_ = out.accessor<int16_t, 1>();

    const uint32_t c_count = 0x10000U;

    for (int i = 0; i < N_sym; ++i)
    {
        const uint64_t span = static_cast<uint64_t>(state.high) - static_cast<uint64_t>(state.low) + 1;
        // always < 0x10000 ???
        const uint16_t count = ((static_cast<uint64_t>(state.value) - static_cast<uint64_t>(state.low) + 1) * c_count - 1) / span;

        const int offset = i * Lp;
        auto sym_i = binsearch(cdf, count, (cdf_t)max_symbol, offset);

        out_[i] = (int16_t)sym_i;

        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : cdf[offset + sym_i + 1];

        state.single_decode(c_low, c_high, in_cache);
    }

#ifdef VERBOSE
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 << std::endl;
#endif

    return out;
}

/** See torchac.py */
class TorchDecoder
{
public:
    TorchDecoder() = default;

    void set_cdf(const at::Tensor &cdf)
    {
        _cdf = cdf.cpu();
    };
    void set_cdf_index(const at::Tensor &cdf, const at::Tensor &cdf_length)
    {
        _cdf = cdf.cpu();
        _cdf_length = cdf_length.cpu();
    };
    void set_string(const std::string &in)
    {
        _state.clear();
        _in_cache = InCacheString(in);
        _state.initialize(_in_cache);
    };
    void set_strings(const std::string &in, const std::string &out_bound_string)
    {
        _state.clear();
        _in_cache = InCacheString(in);
        _state.initialize(_in_cache);
        _outbound_cache = InCacheString(out_bound_string);
    };
    at::Tensor decode_cdf_index(
        const at::Tensor &cdf, /* NLp */
        const at::Tensor &cdf_length,
        const at::Tensor &indexes,
        const std::string &in,
        const std::string &out_bound_string);
    at::Tensor decode_index(
        const at::Tensor &indexes);
    at::Tensor decode_cdf(
        const at::Tensor &cdf);
    at::Tensor _cdf;
    at::Tensor _cdf_length;
    rans_state _state;
    InCacheString _in_cache;
    InCacheString _outbound_cache;
};

at::Tensor TorchDecoder::decode_cdf_index(
    const at::Tensor &cdf, /* NLp */
    const at::Tensor &cdf_length,
    const at::Tensor &indexes,
    const std::string &main_string,
    const std::string &out_bound_string)
{
    rans_state state;
    InCacheString in_cache(main_string);
    state.initialize(in_cache);
    InCacheString outbound_cache(out_bound_string);
    return decode(state, in_cache, outbound_cache, cdf.cpu(), cdf_length.cpu(), indexes.cpu());
}

at::Tensor TorchDecoder::decode_index(
    const at::Tensor &indexes)
{
    return decode(_state, _in_cache, _outbound_cache, _cdf, _cdf_length, indexes.cpu());
}

at::Tensor TorchDecoder::decode_cdf(
    const at::Tensor &cdf)
{
    return decode_cdf_(_state, _in_cache, cdf.cpu());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "torchac";
    py::class_<TorchEncoder>(m, "TorchEncoder")
        .def(py::init<>())
        .def("encode_cdf_index", &TorchEncoder::encode_cdf_index, "Encode from CDF")
        .def("set_cdf", &TorchEncoder::set_cdf, "Set CDF")
        .def("set_cdf_index", &TorchEncoder::set_cdf_index, "Set CDF")
        .def("encode_index", &TorchEncoder::encode_index, "Encode from index")
        .def("encode_cdf", &TorchEncoder::encode_cdf, "Encode from cdf")
        .def("flush", &TorchEncoder::flush, "Flush to string");
    py::class_<TorchDecoder>(m, "TorchDecoder")
        .def(py::init<>())
        .def("decode_cdf_index", &TorchDecoder::decode_cdf_index, "Decode from CDF")
        .def("set_cdf", &TorchDecoder::set_cdf, "Set CDF")
        .def("set_cdf_index", &TorchDecoder::set_cdf_index, "Set CDF")
        .def("set_string", &TorchDecoder::set_string, "Set string")
        .def("set_strings", &TorchDecoder::set_strings, "Set string")
        .def("decode_index", &TorchDecoder::decode_index, "Decode from index")
        .def("decode_cdf", &TorchDecoder::decode_cdf, "Decode from cdf");
#ifdef COMPILE_CUDA
    m.def("cuda_supported", []()
          { return true; });
#else
    m.def("cuda_supported", []()
          { return false; });
#endif
}
