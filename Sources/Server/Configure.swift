import Vapor
import StanfordLlama

// configures your application
public func configure(_ app: Application) async throws {

    let model = try await loadModel()

    let llmController = LLMController(model: model)
    try app.register(collection: llmController)

    try routes(app)
}

func loadModel() async throws -> StanfordLlama.Model {
    let model = StanfordLlama.Model(
        configuration: .init(path: "/Users/steven.prichard/Developer/llamaLibModels/TinyLlama-1.1B-Chat-v1.0-GGUF.gguf"),
        platform: .init(configuration: .init())
    )

    try await model.load()
    return model
}

public func routes(_ app: Application) throws {
    app.get { req async in
        "It works!"
    }
}

public final class LLMController: RouteCollection {
    private var model: StanfordLlama.Model

    public func boot(routes: any Vapor.RoutesBuilder) throws {
        let modelRoutes = routes.grouped("model")
        modelRoutes.post("stanford", use: getStanfordLlamaResponse)
    }

    init(model: StanfordLlama.Model) {
        self.model = model
    }

    func getStanfordLlamaResponse(req: Request) async throws -> ModelResponse {
        let promptRequest = try req.content.decode(PromptRequest.self)

        let responseStream = try await model.respond(to: promptRequest.prompt)

        var response = ""

        for try await partialResponse in responseStream {
            response += partialResponse
        }

        return ModelResponse(
            prompt: promptRequest.prompt,
            response: response
        )
    }
}

public extension LLMController {
    enum Errors: Error {
        case stanfordModelNotLoaded
    }
}


public extension LLMController {
    struct PromptRequest: Codable {
        var prompt: String
    }

    struct ModelResponse: Content {
        var prompt: String
        var response: String
    }
}
