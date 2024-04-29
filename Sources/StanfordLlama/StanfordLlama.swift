@preconcurrency import llama
import Foundation

public struct StanfordLlama {}

public extension StanfordLlama {
    final actor Model: Sendable {
        private let configuration: StanfordLlama.Configuration
        private let platform: Platform
        private var model: OpaquePointer?
        private var context: OpaquePointer?
        /// A task managing the  output generation.
        private var generationTask: Task<(), any Error>?

        public init(
            configuration: StanfordLlama.Configuration,
            platform: StanfordLlama.Platform
        ) {
            self.configuration = configuration
            self.platform = platform
        }

        public func load() throws {
            guard let model = llama_load_model_from_file(
                configuration.path,
                configuration.parameters.llamaCppRepresentation
            ) else {
                throw Errors.failedToLoad
            }

            guard configuration.contextParameters.contextWindowSize <= llama_n_ctx_train(model) else {
                throw Errors.contextWindowSizeTooLarge
            }

            self.model = model
        }

        public func respond(to prompt: String) async throws -> AsyncThrowingStream<String, Error> {
            try await platform.exclusiveAccess()

            let (stream, continuation) = AsyncThrowingStream.makeStream(of: String.self)

            generationTask = Task(priority: platform.configuration.taskPriority) {
                defer {
                    Task {
                        await platform.signal()
                    }
                }

                // Execute the inference
                try await generate(prompt: prompt, with: continuation)
            }

            return stream
        }

        // TODO: should we lock access to the context? Or does InferenceActor save us
        private func generate(prompt: String, with continuation: AsyncThrowingStream<String, Error>.Continuation) async throws {
            let context = try createContextIfNeeded()

            guard configuration.parameters.maxOutputLength <= configuration.contextParameters.contextWindowSize else {
                await finishGenerationWithError(Errors.maxOutputSizeTooLarge, on: continuation)
                return
            }

            let tokens = self.tokenize(input: prompt, with: context)

            guard await !checkCancellation(on: continuation) else {
                return
            }

            guard tokens.count <= configuration.contextParameters.contextWindowSize - 4 else {
                await finishGenerationWithError(Errors.inputPromptTooLarge, on: continuation)
                return
            }

            // Clear the KV Cache, to make room for the incoming prompt
            llama_kv_cache_clear(context)

            var batch = llama_batch_init(Int32(tokens.count), 0, 1)
            defer { llama_batch_free(batch) }

            // Evaluate the initial prompt
            for (tokenIndex, token) in tokens.enumerated() {
                llama_batch_add(&batch, token, Int32(tokenIndex), getLlamaSeqIdVector(), false)
            }
            // llama_decode will output logits only for the last token of the prompt
            batch.logits[Int(batch.n_tokens) - 1] = 1

            guard await !checkCancellation(on: continuation) else {
                return
            }

            if llama_decode(context, batch) != 0 {
                await finishGenerationWithError(Errors.failedToDecodeBatch, on: continuation)
                return
            }

            guard await !checkCancellation(on: continuation) else {
                return
            }

            // Batch already includes tokens from the input prompt
            var batchTokenIndex = batch.n_tokens
            var decodedTokens = 0

            while decodedTokens <= configuration.parameters.maxOutputLength {
                guard await !checkCancellation(on: continuation) else {
                    return
                }

                let nextTokenID = sample(batchSize: batch.n_tokens, with: context)

                // Finish generation once EOS Token is present, max output length of answer is reached, or context window is reached
                if nextTokenID == llama_token_eos(model) ||
                    decodedTokens == configuration.parameters.maxOutputLength ||
                    batchTokenIndex == configuration.contextParameters.contextWindowSize {
                    print("🟢 reached stopping condition")
                    continuation.finish()
                }

                var nextStringPiece = String(llama_token_to_piece(context, nextTokenID))
                // As first character is sometimes randomly prefixed by a single space (even though prompt has an additional character)
                if decodedTokens == 0 && nextStringPiece.starts(with: " ") {
                    nextStringPiece = String(nextStringPiece.dropFirst())
                }

                // print("ℹ️ Yielded Token: \(nextStringPiece)")

                if nextTokenID != 0 {
                    continuation.yield(nextStringPiece)
                }

                // Could inject piece into context here

                // Prepare the next batch
                llama_batch_clear(&batch)

                // Add generated token to next generation round
                llama_batch_add(&batch, nextTokenID, batchTokenIndex, getLlamaSeqIdVector(), true)

                decodedTokens += 1
                batchTokenIndex += 1

                // Evaluate the current batch with the transformer model
                // = 0 Success, > 0 Warning, < 0 Error
                let decodedOutput = llama_decode(context, batch)
                if decodedOutput != 0 {
                    await finishGenerationWithError(Errors.failedToDecodeBatch, on: continuation)
                    return
                }
            }

            llama_print_timings(context)

            continuation.finish()
            print("✅ Local LLM completed an inference")
        }

        private func checkCancellation(on continuation: AsyncThrowingStream<String, Error>.Continuation) async -> Bool {
            if Task.isCancelled {
                await finishGenerationWithError(CancellationError(), on: continuation)
                return true
            }

            return false
        }

        private func sample(batchSize: Int32, with context: OpaquePointer) -> llama_token {
            let nVocab = llama_n_vocab(model)
            let logits = llama_get_logits_ith(context, batchSize - 1)
            var candidates: [llama_token_data] = .init(repeating: llama_token_data(), count: Int(nVocab))

            for tokenID in 0 ..< nVocab {
                candidates.append(llama_token_data(id: tokenID, logit: logits?[Int(tokenID)] ?? 0, p: 0.0))
            }

            var candidatesProbability: llama_token_data_array = .init(
                data: candidates.withUnsafeMutableBytes { $0.baseAddress?.assumingMemoryBound(to: llama_token_data.self) }, // the same as &candidates
                size: candidates.count,
                sorted: false
            )

            let minimumKeepCount = Int(max(1, configuration.samplingParameters.outputProbabilities))
            llama_sample_top_k(context, &candidatesProbability, configuration.samplingParameters.topK, minimumKeepCount)
            llama_sample_top_p(context, &candidatesProbability, configuration.samplingParameters.topP, minimumKeepCount)
            llama_sample_temp(context, &candidatesProbability, configuration.samplingParameters.temperature)

            return llama_sample_token(context, &candidatesProbability)
        }

        private func createContextIfNeeded() throws -> OpaquePointer {
            guard let model else {
                throw Errors.modelNotLoaded
            }

            if let context {
                return context
            } else {
                guard let context = llama_new_context_with_model(model, configuration.contextParameters.llamaCppRepresentation) else {
                    throw Errors.failedToCreateContext
                }
                self.context = context
                return context
            }
        }

        func tokenize(input: String, with context: OpaquePointer) -> [llama_token] {
            var tokens: [llama_token] = .init(
                llama_tokenize_with_context(context, std.string(input), configuration.parameters.addBosToken, true)
            )

            // Truncate tokens if there wouldn't be enough context size for the generated output
            if tokens.count > Int(configuration.contextParameters.contextWindowSize) - configuration.parameters.maxOutputLength {
                tokens = Array(tokens.suffix(Int(configuration.contextParameters.contextWindowSize) - configuration.parameters.maxOutputLength))
            }

            if tokens.isEmpty {
                tokens.append(llama_token_bos(self.model))
                print("BOS token inserted")
            }

            return tokens
        }
    }
}

private extension StanfordLlama.Model {
    func finishGenerationWithError<E: Error>(_ error: E, on continuation: AsyncThrowingStream<String, Error>.Continuation) async {
        continuation.finish(throwing: error)
    }
}

public extension StanfordLlama.Model {
    enum Errors: Error {
        case failedToLoad
        case modelNotLoaded
        case contextWindowSizeTooLarge
        case failedToCreateContext
        case maxOutputSizeTooLarge
        case inputPromptTooLarge
        case failedToDecodeBatch
    }
}

public extension StanfordLlama {
    actor Platform {
        private let semaphore = AsyncSemaphore(limit: 1)
        let configuration: Configuration

        public init(configuration: Configuration) {
            self.configuration = configuration
        }

        public init() {
            self.init(configuration: .init())
        }

        deinit {
            llama_backend_free()
        }

        func initialize() {
            llama_backend_init()
            llama_numa_init(configuration.memoryAccess.wrappedValue)
        }

        nonisolated func exclusiveAccess() async throws {
            try await semaphore.withCheckingCancelation()
            // Stanford calls a MainActor.run to update state. Maybe we could do that but for the inference actor?
        }

        nonisolated func signal() async {
            semaphore.signal()
        }
    }
}

public extension StanfordLlama {
    struct Configuration: Sendable {
        let path: String
        let parameters: Parameters
        let contextParameters: ContextParameters
        let samplingParameters: SamplingParameters
        // TODO: May need a more complex input type than String. For handling role & content
        // Defaults to no preprocessing
        let preprocessor: (@Sendable (String) -> String) = { input in return input }

        public init(
            path: String,
            parameters: Parameters = .init(),
            contextParameters: ContextParameters = .init(),
            samplingParameters: SamplingParameters = .init()
        ) {
            self.path = path
            self.parameters = parameters
            self.contextParameters = contextParameters
            self.samplingParameters = samplingParameters
        }
    }
}

public extension StanfordLlama.Configuration {
    struct Parameters: Sendable {
        /// The to-be-used system prompt of the LLM
        let systemPrompt: String?
        /// Indicates the maximum output length generated by the LLM.
        let maxOutputLength: Int
        /// Indicates whether the BOS token is added by the LLM. If `nil`, the default from the model itself is taken.
        let addBosToken: Bool

        /// Wrapped C struct from the llama.cpp library, later-on passed to the LLM
        private var wrapped: llama_model_params

        /// Model parameters in llama.cpp's low-level C representation
        var llamaCppRepresentation: llama_model_params { wrapped }

        /// Number of layers to store in VRAM
        /// - Note: On iOS simulators, this property has to be set to 0 (which is automatically done by the library).
        var gpuLayerCount: Int32 {
            get { wrapped.n_gpu_layers }
            set { wrapped.n_gpu_layers = newValue }
        }

        /// Indicates the GPU that is used for scratch and small tensors.
        var mainGpu: Int32 {
            get { wrapped.main_gpu }
            set { wrapped.main_gpu = newValue }
        }

        /// Indicates how to split layers across multiple GPUs.
        var tensorSplit: UnsafePointer<Float>? {
            get { wrapped.tensor_split }
            set { wrapped.tensor_split = newValue }
        }

        /// Context pointer that is passed to the progress callback
        var progressCallbackUserData: UnsafeMutableRawPointer? {
            get { wrapped.progress_callback_user_data }
            set { wrapped.progress_callback_user_data = newValue }
        }

        /// Indicates wether booleans should be kept together to avoid misalignment during copy-by-value.
        var vocabOnly: Bool {
            get { wrapped.vocab_only }
            set { wrapped.vocab_only = newValue }
        }

        /// Indicates if mmap should be used.
        var useMmap: Bool {
            get { wrapped.use_mmap }
            set { wrapped.use_mmap = newValue}
        }

        /// Forces the system to keep model in RAM.
        var useMlock: Bool {
            get { wrapped.use_mlock }
            set { wrapped.use_mlock = newValue}
        }

        /// Creates the ``Parameters`` which wrap the underlying llama.cpp `llama_model_params` C struct.
        /// - Parameters:
        ///   - systemPrompt: The to-be-used system prompt of the LLM enabling fine-tuning of the LLMs behaviour. Defaults to the regular default chat-based LLM system prompt.
        ///   - maxOutputLength: The maximum output length generated by the Spezi LLM, defaults to `512`.
        ///   - addBosToken: Indicates wether the BOS token is added by the Spezi LLM, defaults to `false`.
        ///   - gpuLayerCount: Number of layers to store in VRAM, defaults to `1`, meaning Apple's `Metal` framework is enabled.
        ///   - mainGpu: GPU that is used for scratch and small tensors, defaults to `0` representing the main GPU.
        ///   - tensorSplit: Split layers across multiple GPUs, defaults to `nil`, meaning no split.
        ///   - progressCallback: Progress callback called with a progress value between 0 and 1, defaults to `nil`.
        ///   - progressCallbackUserData: Context pointer that is passed to the progress callback, defaults to `nil`.
        ///   - vocabOnly: Indicates wether booleans should be kept together to avoid misalignment during copy-by-value., defaults to `false`.
        ///   - useMmap: Indicates if mmap should be used., defaults to `true`.
        ///   - useMlock: Forces the system to keep model in RAM, defaults to `false`.
        public init(
            systemPrompt: String? = Defaults.systemPrompt,
            maxOutputLength: Int = Defaults.maxOutputLength,
            addBosToken: Bool = Defaults.addBosToken,
            gpuLayerCount: Int32 = Defaults.gpuLayerCount,
            mainGpu: Int32 = Defaults.mainGpu,
            tensorSplit: UnsafePointer<Float>? = Defaults.tensorSplit,
            progressCallbackUserData: UnsafeMutableRawPointer? = Defaults.progressCallbackUserData,
            vocabOnly: Bool = Defaults.vocabOnly,
            useMmap: Bool = Defaults.useMmap,
            useMlock: Bool = Defaults.useMlock
        ) {
            self.wrapped = llama_model_default_params()

            self.systemPrompt = systemPrompt
            self.maxOutputLength = maxOutputLength
            self.addBosToken = addBosToken

            /// Overwrite `gpuLayerCount` in case of a simulator target environment
            #if targetEnvironment(simulator)
            self.gpuLayerCount = 0     // Disable Metal on simulator as crash otherwise
            #else
            self.gpuLayerCount = gpuLayerCount
            #endif
            self.mainGpu = mainGpu
            self.tensorSplit = tensorSplit
            self.progressCallbackUserData = progressCallbackUserData
            self.vocabOnly = vocabOnly
            self.useMmap = useMmap
            self.useMlock = useMlock
        }
    }
}

public extension StanfordLlama.Configuration.Parameters {
    enum Defaults {
        public static let systemPrompt: String = { "" }()
        public static let maxOutputLength: Int = { 512 }()
        public static let addBosToken: Bool = { false }()
        public static let gpuLayerCount: Int32 = { 1 }()
        public static let mainGpu: Int32 = { 1 }()
        public static let tensorSplit: UnsafePointer<Float>? = { nil }()
        public static let progressCallbackUserData: UnsafeMutableRawPointer? = { nil }()
        public static let vocabOnly: Bool = { false }()
        public static let useMmap: Bool = { true }()
        public static let useMlock: Bool = { false }()
    }
}


public extension StanfordLlama.Configuration {
    struct ContextParameters: Sendable {
        /// Wrapped C struct from the llama.cpp library, later-on passed to the LLM
        private var wrapped: llama_context_params

        /// Context parameters in llama.cpp's low-level C representation
        var llamaCppRepresentation: llama_context_params {
            wrapped
        }

        /// RNG seed of the LLM
        var seed: UInt32 {
            get { wrapped.seed }
            set { wrapped.seed = newValue }
        }

        /// Context window size in tokens (0 = take default window size from model)
        var contextWindowSize: UInt32 {
            get { wrapped.n_ctx }
            set { wrapped.n_ctx = newValue }
        }

        /// Maximum batch size during prompt processing
        var batchSize: UInt32 {
            get { wrapped.n_batch }
            set { wrapped.n_batch = newValue }
        }

        /// Number of threads used by LLM for generation of output
        var threadCount: UInt32 {
            get { wrapped.n_threads }
            set { wrapped.n_threads = newValue }
        }

        /// Number of threads used by LLM for batch processing
        var batchThreadCount: UInt32 {
            get { wrapped.n_threads_batch }
            set { wrapped.n_threads_batch = newValue }
        }

        /// RoPE base frequency (0 = take default from model)
        var ropeFreqBase: Float {
            get { wrapped.rope_freq_base }
            set { wrapped.rope_freq_base = newValue }
        }

        /// RoPE frequency scaling factor (0 = take default from model)
        var ropeFreqScale: Float {
            get { wrapped.rope_freq_scale }
            set { wrapped.rope_freq_scale = newValue }
        }

        /// If `true`, offload the KQV ops (including the KV cache) to GPU
        var offloadKQV: Bool {
            get { wrapped.offload_kqv }
            set { wrapped.offload_kqv = newValue }
        }

        /// ``GGMLType`` of the key of the KV cache
        var kvKeyType: GGMLType {
            get { GGMLType(rawValue: wrapped.type_k.rawValue) ?? .f16 }
            set { wrapped.type_k = ggml_type(rawValue: newValue.rawValue) }
        }

        /// ``GGMLType`` of the value of the KV cache
        var kvValueType: GGMLType {
            get { GGMLType(rawValue: wrapped.type_v.rawValue) ?? .f16 }
            set { wrapped.type_v = ggml_type(rawValue: newValue.rawValue) }
        }

        /// If `true`, the (deprecated) `llama_eval()` call computes all logits, not just the last one
        var computeAllLogits: Bool {
            get { wrapped.logits_all }
            set { wrapped.logits_all = newValue }
        }

        /// Creates the ``LLMLocalContextParameters`` which wrap the underlying llama.cpp `llama_context_params` C struct.
        /// Is passed to the underlying llama.cpp model in order to configure the context of the LLM.
        ///
        /// - Parameters:
        ///   - seed: RNG seed of the LLM, defaults to `4294967295` (which represents a random seed).
        ///   - contextWindowSize: Context window size in tokens, defaults to `1024`.
        ///   - batchSize: Maximum batch size during prompt processing, defaults to `1024` tokens.
        ///   - threadCount: Number of threads used by LLM for generation of output, defaults to the processor count of the device.
        ///   - batchThreadCount: Number of threads used by LLM for batch processing, defaults to the processor count of the device.
        ///   - ropeFreqBase: RoPE base frequency, defaults to `0` indicating the default from model.
        ///   - ropeFreqScale: RoPE frequency scaling factor, defaults to `0` indicating the default from model.
        ///   - offloadKQV: Offloads the KQV ops (including the KV cache) to GPU, defaults to `true`.
        ///   - kvKeyType: ``GGMLType`` of the key of the KV cache, defaults to ``GGMLType/f16``.
        ///   - kvValueType: ``GGMLType`` of the value of the KV cache, defaults to ``GGMLType/f16``.
        ///   - computeAllLogits: `llama_eval()` call computes all logits, not just the last one. Defaults to `false`.
        public init(
            seed: UInt32 = 4294967295,
            contextWindowSize: UInt32 = 1024,
            batchSize: UInt32 = 1024,
            threadCount: UInt32 = .init(ProcessInfo.processInfo.processorCount),
            threadCountBatch: UInt32 = .init(ProcessInfo.processInfo.processorCount),
            ropeFreqBase: Float = 0.0,
            ropeFreqScale: Float = 0.0,
            useMulMatQKernels: Bool = true,
            offloadKQV: Bool = true,
            kvKeyType: GGMLType = .f16,
            kvValueType: GGMLType = .f16,
            computeAllLogits: Bool = false
        ) {
            self.wrapped = llama_context_default_params()
            self.seed = seed
            self.contextWindowSize = contextWindowSize
            self.batchSize = batchSize
            self.threadCount = threadCount
            self.batchThreadCount = threadCountBatch
            self.ropeFreqBase = ropeFreqBase
            self.ropeFreqScale = ropeFreqScale
            self.offloadKQV = offloadKQV
            self.kvKeyType = kvKeyType
            self.kvValueType = kvValueType
            self.computeAllLogits = computeAllLogits
        }
    }
}

public extension StanfordLlama.Configuration.ContextParameters {
    /// Swift representation of the `ggml_type` of llama.cpp, indicating data types within KV caches.
    enum GGMLType: UInt32, Sendable {
        case f32 = 0
        case f16
        case q4_0
        case q4_1
        case q5_0 = 6
        case q5_1
        case q8_0
        case q8_1
        /// k-quantizations
        case q2_k
        case q3_k
        case q4_k
        case q5_k
        case q6_k
        case q8_k
        case iq2_xxs
        case iq2_xs
        case i8
        case i16
        case i32
    }
}

public extension StanfordLlama.Configuration.ContextParameters {
    enum Defaults {
        public static let seed: UInt32 = { 20 }()
        public static let contextWindowSize: UInt32 = { 1024 }()
        public static let batchSize: UInt32 = { 1024 }()
        public static let threadCount: UInt32 = { .init(ProcessInfo.processInfo.processorCount) }()
        public static let threadCountBatch: UInt32 = { .init(ProcessInfo.processInfo.processorCount) }()
        public static let ropeFreqBase: Float = { 0.0 }()
        public static let ropeFreqScale: Float = { 0.0 }()
        public static let useMulMatQKernels: Bool = { true }()
        public static let offloadKQV: Bool = { true }()
        public static let kvKeyType: GGMLType = { .f16 }()
        public static let kvValueType: GGMLType = { .f16 }()
        public static let computeAllLogits: Bool = { false }()
    }
}

public extension StanfordLlama.Configuration {
    struct SamplingParameters: Sendable {
        /// Top-K Sampling: K most likely next words (<= 0 to use vocab size).
        var topK: Int32
        /// Top-p Sampling: Smallest possible set of words whose cumulative probability exceeds the probability p (1.0 = disabled).
        var topP: Float
        /// Temperature Sampling: A higher value indicates more creativity of the model but also more hallucinations.
        var temperature: Float
        /// If greater than 0, output the probabilities of top n_probs tokens.
        var outputProbabilities: Int

        public init(
            topK: Int32 = Defaults.topK,
            topP: Float = Defaults.topP,
            temperature: Float = Defaults.temperature,
            outputProbabilities: Int = Defaults.outputProbabilities
        ) {
            self.topK = topK
            self.topP = topP
            self.temperature = temperature
            self.outputProbabilities = outputProbabilities
        }
    }
}

public extension StanfordLlama.Configuration.SamplingParameters {
    enum Defaults {
        public static let topK: Int32 = { 40 }()
        public static let topP: Float = { 0.95 }()
        public static let temperature: Float = { 0.8 }()
        public static let outputProbabilities: Int = { 0 }()
    }
}

public extension StanfordLlama.Platform {
    struct Configuration: Sendable {
        /// Wrapper around the `ggml_numa_strategy` type of llama.cpp, indicating the non-unified memory access configuration of the device.
        public enum NonUniformMemoryAccess: UInt32, Sendable {
            case disabled
            case distributed
            case isolated
            case numaCtl
            case mirror
            case count

            var wrappedValue: ggml_numa_strategy {
                .init(rawValue: self.rawValue)
            }
        }

        var memoryAccess: NonUniformMemoryAccess
        var taskPriority: TaskPriority

        public init(
            memoryAccess: NonUniformMemoryAccess = .disabled,
            taskPriority: TaskPriority = .userInitiated
        ) {
            self.memoryAccess = memoryAccess
            self.taskPriority = taskPriority
        }
    }
}
