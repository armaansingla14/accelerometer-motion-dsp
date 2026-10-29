#include "motiondsp/window_io.hpp"

#include <fstream>
#include <stdexcept>

namespace motiondsp {

WindowSet load_windows(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::size_t n = 0;
    WindowSet ws;
    if (!(in >> n >> ws.window >> ws.channels)) throw std::runtime_error("bad header in " + path);
    ws.bias.resize(n * ws.channels);
    ws.samples.resize(n * ws.window * ws.channels);
    for (std::size_t w = 0; w < n; ++w) {
        for (std::size_t c = 0; c < ws.channels; ++c)
            if (!(in >> ws.bias[w * ws.channels + c])) throw std::runtime_error("truncated bias");
        double* dst = &ws.samples[w * ws.window * ws.channels];
        for (std::size_t i = 0; i < ws.window * ws.channels; ++i)
            if (!(in >> dst[i])) throw std::runtime_error("truncated samples");
    }
    return ws;
}

}  // namespace motiondsp
