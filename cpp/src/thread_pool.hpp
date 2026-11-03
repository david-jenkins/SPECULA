#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace specula_cpp {

// Persistent fork-join thread pool.
//
// run(n, task) calls task(0) ... task(n-1) on the worker threads and on the
// calling thread, and returns when all calls are done. Calls made from inside
// a task, or while another thread is already using the pool, are executed
// serially by the calling thread, so the pool can never deadlock.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads) { start(n_threads); }
    ~ThreadPool() { stop(); }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    int size() const { return n_threads_; }

    void resize(int n_threads) {
        std::lock_guard<std::mutex> run_lock(run_mutex_);
        stop();
        start(n_threads);
    }

    void run(int n_tasks, const std::function<void(int)> &task) {
        if (n_tasks <= 0)
            return;
        std::unique_lock<std::mutex> run_lock(run_mutex_, std::defer_lock);
        if (n_tasks == 1 || workers_.empty() || in_task() || !run_lock.try_lock()) {
            for (int i = 0; i < n_tasks; ++i)
                task(i);
            return;
        }
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // Workers that woke up late for the previous job must be done
            // before the shared state is reset for this one.
            done_cv_.wait(lock, [this] { return active_ == 0; });
            task_ = &task;
            n_tasks_ = n_tasks;
            next_.store(0);
            ++generation_;
        }
        cv_.notify_all();
        in_task() = true;
        work(&task, n_tasks);
        in_task() = false;
        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [this] { return active_ == 0; });
            std::swap(error, error_);
        }
        if (error)
            std::rethrow_exception(error);
    }

private:
    static bool &in_task() {
        thread_local bool flag = false;
        return flag;
    }

    void start(int n_threads) {
        n_threads_ = std::max(1, n_threads);
        stop_ = false;
        for (int i = 1; i < n_threads_; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &w : workers_)
            w.join();
        workers_.clear();
    }

    // Late workers can find no task left: they never call task, which may be gone
    void work(const std::function<void(int)> *task, int n_tasks) {
        for (int i = next_.fetch_add(1); i < n_tasks; i = next_.fetch_add(1)) {
            try {
                (*task)(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!error_)
                    error_ = std::current_exception();
                next_.store(n_tasks);  // hand out no more tasks
            }
        }
    }

    void worker_loop() {
        in_task() = true;
        std::uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)> *task;
            int n_tasks;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_)
                    return;
                seen = generation_;
                task = task_;
                n_tasks = n_tasks_;
                ++active_;
            }
            work(task, n_tasks);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_;
            }
            done_cv_.notify_all();
        }
    }

    int n_threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex run_mutex_;  // one job at a time
    std::mutex mutex_;      // protects the job state below
    std::condition_variable cv_, done_cv_;
    const std::function<void(int)> *task_ = nullptr;
    std::exception_ptr error_;
    int n_tasks_ = 0;
    int active_ = 0;
    std::uint64_t generation_ = 0;
    bool stop_ = false;
    std::atomic<int> next_{0};
};

ThreadPool &pool();

// Split [0, n) in contiguous ranges of at least `grain` items and call
// f(begin, end) on each range, in parallel.
template <class F>
void parallel_for(std::ptrdiff_t n, std::ptrdiff_t grain, F &&f) {
    if (n <= 0)
        return;
    grain = std::max<std::ptrdiff_t>(1, grain);
    std::ptrdiff_t n_tasks = std::min<std::ptrdiff_t>(pool().size(), (n + grain - 1) / grain);
    if (n_tasks <= 1) {
        f(std::ptrdiff_t(0), n);
        return;
    }
    std::ptrdiff_t chunk = (n + n_tasks - 1) / n_tasks;
    pool().run(static_cast<int>(n_tasks), [&](int t) {
        std::ptrdiff_t begin = t * chunk;
        std::ptrdiff_t end = std::min(n, begin + chunk);
        if (begin < end)
            f(begin, end);
    });
}

}  // namespace specula_cpp
