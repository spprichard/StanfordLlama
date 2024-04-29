// https://github.com/StanfordSpezi/SpeziFoundation/blob/main/Sources/SpeziFoundation/Semaphore/AsyncSemaphore.swift
// Based of the Stanford implementation.
// Used to control access the to llama.cpp, which should eliminate memory access errors


import Foundation

public extension StanfordLlama {
    final class AsyncSemaphore: @unchecked Sendable {
        private enum Suspension {
            case cancelable(UnsafeContinuation<Void, Error>)
            case regular(UnsafeContinuation<Void, Never>)

            func resume() {
                switch self {
                    case let .cancelable(unsafeContinuation):
                        unsafeContinuation.resume()

                    case let .regular(unsafeContinuation):
                        unsafeContinuation.resume()
                }
            }
        }

        private struct SuspendedTask: Identifiable {
            let id: UUID
            let suspension: Suspension
        }

        private var value: Int
        private var suspendedTasks: [SuspendedTask] = []
        private let nsLock = NSLock()

        /// Initializes a new semaphore with a given concurrency limit.
        /// - Parameter value: The maximum number of concurrent accesses allowed. Must be non-negative.
        public init(limit: Int) {
            precondition(limit >= 0)
            self.value = limit
        }

        /// Decreases the semaphore count and waits if the count is less than zero.
        /// Use this method when access to a resource should be awaited without the possibility of cancellation.
        public func wait() async {
            lock()

            value -= 1
            if value >= 0 {
                unlock()
                return
            }

            await withUnsafeContinuation { continuation in
                suspendedTasks.append(SuspendedTask(id: UUID(), suspension: .regular(continuation)))
                unlock()
            }
        }

        /// Decreases the semaphore count and throws a `CancellationError` if the current `Task` is cancelled.
        /// This method allows the `Task` calling ``waitCheckingCancellation()`` to be cancelled while waiting, 
        /// throwing a `CancellationError` if the `Task` is cancelled before it can proceed.
        /// - Throws: `CancellationError` if the task is cancelled while waiting.
        public func withCheckingCancelation() async throws {
            try Task.checkCancellation()

            lock()

            do {
                // Check if task was canceled while obtaining lock
                try Task.checkCancellation()
            } catch {
                unlock()
                throw error
            }

            value -= 1
            if value >= 0 {
                unlock()
                return
            }

            let id = UUID()

            try await withTaskCancellationHandler {
                try await withUnsafeThrowingContinuation { (continuation: UnsafeContinuation<Void, Error>) in
                    if Task.isCancelled {
                        value += 1 // restore value
                        unlock()
                        continuation.resume(throwing: CancellationError())
                    } else {
                        suspendedTasks.append(SuspendedTask(id: id, suspension: .cancelable(continuation)))
                        unlock()
                    }
                }
            } onCancel: {
                lock()
                value += 1
                guard let index = self.suspendedTasks.firstIndex(where: { $0.id == id }) else {
                    preconditionFailure("Inconsistent internal state reached")
                }

                let task = suspendedTasks[index]
                suspendedTasks.remove(at: index)

                unlock()
                switch task.suspension {
                    case .regular:
                        preconditionFailure("Tried to cancel a task that was not cancellable!")
                    case let .cancelable(continuation):
                        continuation.resume(throwing: CancellationError())
                }
            }
        }

        /// Signals the semaphore, allowing one waiting task to proceed.
        /// If there are `Task`s waiting for access, calling this method will resume one of them.
        /// - Returns: `true` if a task was resumed, `false` otherwise.
        @discardableResult
        public func signal() -> Bool {
            lock()

            value += 1
            guard let firstTask = suspendedTasks.first else {
                unlock()
                return false
            }

            suspendedTasks.removeFirst()
            unlock()

            firstTask.suspension.resume()
            return true
        }

        /// Signals the semaphore, allowing all waiting `Task`s to proceed.
        /// This method resumes all `Task`s that are currently waiting for access.
        public func signalAll() {
            lock()

            value += suspendedTasks.count
            let tasks = suspendedTasks
            self.suspendedTasks.removeAll()

            unlock()

            for task in tasks {
                task.suspension.resume()
            }
        }

        /// Cancels all waiting `Task`s that can be cancelled.
        /// This method attempts to cancel all `Task`s that are currently waiting and support cancellation. `Task`s that do not support cancellation will cause a runtime error.
        /// - Warning: Will trigger a runtime error if it attempts to cancel `Task`s that are not cancellable.
        public func cancelAll() {
            lock()

            value += suspendedTasks.count
            let tasks = suspendedTasks
            self.suspendedTasks.removeAll()

            unlock()

            for task in tasks {
                switch task.suspension {
                    case let .cancelable(unsafeContinuation):
                        unsafeContinuation.resume(throwing: CancellationError())
                    case .regular:
                        // TODO: I wonder if we could just ignore this case...
                        preconditionFailure("Tried to cancel a task that was not cancellable!")
                }
            }
        }

        private func lock() {
            nsLock.lock()
        }

        private func unlock() {
            nsLock.unlock()
        }
    }
}
