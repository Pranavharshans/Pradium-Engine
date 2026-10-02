#include "graph_replay_plan.h"
#include <cassert>
#include <random>
#include <vector>

int main()
{
    GraphReplayPlan plan;
    std::vector<int> sites{1, 2, 0, 1, 2, 0, 3, 4, 0};
    std::vector<int> args{1, 1, 3, 4};
    auto bind = [&]() -> const std::vector<std::size_t>& {
        return plan.bind(sites.size(), [&](std::size_t i) { return sites[i]; },
                         args.size(), [&](std::size_t i) { return args[i]; });
    };
    assert((bind() == std::vector<std::size_t>{0, 3, 6, 7}));
    assert((bind() == std::vector<std::size_t>{0, 3, 6, 7}));
    args = {1, 0, 1, 0, 3};
    assert((bind() == std::vector<std::size_t>{0, 2, 3, 5, 6}));
    args = {99};
    bool rejected = false;
    try { bind(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    args = {1, 0, 1, 0, 3};
    assert((bind() == std::vector<std::size_t>{0, 2, 3, 5, 6}));
    args.clear();
    assert(bind().empty());

    // Compare cached/rebound layouts to the original ordered subsequence scan,
    // with many repeated IDs and omitted static sites.
    std::mt19937 random(42);
    for (int trial = 0; trial < 10000; ++trial)
    {
        sites.clear(); args.clear();
        GraphReplayPlan trial_plan;
        for (int i = 0; i < 64; ++i) sites.push_back(random() % 8);
        for (int i = 0; i < 64; ++i)
            if (random() % 3 == 0) args.push_back(sites[i]);
        std::vector<std::size_t> reference;
        std::size_t cursor = 0;
        for (int id : args)
        {
            while (sites[cursor] != id) ++cursor;
            reference.push_back(cursor++);
        }
        for (int replay = 0; replay < 3; ++replay)
        {
            const auto& actual = trial_plan.bind(
                sites.size(), [&](std::size_t i) { return sites[i]; },
                args.size(), [&](std::size_t i) { return args[i]; });
            assert(actual == reference);
        }
    }
}
