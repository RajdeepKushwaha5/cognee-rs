package ai.cognee;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.cognee.internal.Json;
import com.fasterxml.jackson.databind.JsonNode;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletionException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CognifyOptionsTest {
    private static final String SCHEMA =
            "{\"title\":\"Graph\",\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}}}";

    @Test
    void graphSchemaAndChunkAttachmentSerialize() {
        JsonNode n = Json.tree(new CognifyOptions().graphSchema(SCHEMA).chunkAttachment("all").toJson());
        // JSON text crosses as a string; the native side parses it.
        assertEquals(SCHEMA, n.path("graphSchema").asText());
        assertEquals("all", n.path("chunkAttachment").asText());

        JsonNode m = Json.tree(new CognifyOptions().graphSchema(Map.of("title", "Graph")).toJson());
        assertEquals("Graph", m.path("graphSchema").path("title").asText());
    }

    @Test
    void graphSpecSerializes() {
        String spec = "{\"entities\":[{\"name\":\"Person\"}]}";
        JsonNode n = Json.tree(new CognifyOptions().graphSpec(spec).toJson());
        assertEquals(spec, n.path("graphSpec").asText());

        JsonNode m = Json.tree(new CognifyOptions()
                .graphSpec(Map.of("entities", java.util.List.of(Map.of("name", "Person"))))
                .toJson());
        assertEquals("Person", m.path("graphSpec").path("entities").path(0).path("name").asText());
    }

    @Test
    void invalidGraphSpecFailsTheCall(@TempDir Path dir) {
        try (Cognee cognee = new Cognee(TestConfig.underTempDir(dir))) {
            CompletionException e = assertThrows(
                    CompletionException.class,
                    () -> cognee.cognify(
                                    "missing",
                                    new CognifyOptions()
                                            .graphSpec("{\"entities\":[{\"name\":\"P\"},{\"name\":\"P\"}]}"))
                            .join());
            CogneeException cause = assertInstanceOf(CogneeException.class, e.getCause());
            assertEquals("VALIDATION_ERROR", cause.code());
            assertTrue(
                    e.getCause().getMessage().contains("Entity names must be unique."),
                    e.getCause().getMessage());
        }
    }

    @Test
    void invalidChunkAttachmentFailsTheCall(@TempDir Path dir) {
        try (Cognee cognee = new Cognee(TestConfig.underTempDir(dir))) {
            CompletionException e = assertThrows(
                    CompletionException.class,
                    () -> cognee.cognify(
                                    "missing",
                                    new CognifyOptions().graphSchema(SCHEMA).chunkAttachment("everything"))
                            .join());
            CogneeException cause = assertInstanceOf(CogneeException.class, e.getCause());
            assertEquals("VALIDATION_ERROR", cause.code());
            assertTrue(e.getCause().getMessage().contains("chunk_attachment"), e.getCause().getMessage());
        }
    }
}
