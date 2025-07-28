import json
from fastapi.testclient import TestClient
from markov import MarkovChain
from main import app

# training data
with open("reasons.json", "r") as f:
    TRAINING_PHRASES = json.load(f)

client = TestClient(app)


class TestMarkovChain:
    def test_markov_chain_initialization(self):
        chain = MarkovChain(order=2)
        assert chain.order == 2
        assert len(chain.chain) == 0
        assert len(chain.start_words) == 0
        assert not chain.is_trained()

    def test_markov_chain_training(self):
        chain = MarkovChain(order=2)
        test_phrases = [
            "Hello world friend",
            "Hello there friend",
            "World peace now"
        ]
        chain.train(test_phrases)
        assert chain.is_trained()
        assert len(chain.start_words) > 0
        assert len(chain.chain) > 0

    def test_markov_chain_generation(self):
        chain = MarkovChain(order=2)
        chain.train(TRAINING_PHRASES)
        generated = chain.generate(max_length=10)
        assert isinstance(generated, str)
        assert len(generated) > 0
        assert len(generated.split()) <= 10

    def test_markov_chain_generation_without_training(self):
        chain = MarkovChain(order=2)
        generated = chain.generate()
        assert generated == "No."

    def test_markov_chain_training_with_short_phrases(self):
        chain = MarkovChain(order=3)
        short_phrases = ["No", "Nope"]  # Shorter than order
        chain.train(short_phrases)
        assert not chain.is_trained()


class TestAPI:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "description" in data
        assert "endpoints" in data

    def test_no_endpoint(self):
        response = client.get("/no")
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_multiple_nos_endpoint_default(self):
        response = client.get("/nos")
        assert response.status_code == 200
        data = response.json()
        assert "responses" in data
        assert "count" in data
        assert data["count"] == 5
        assert len(data["responses"]) == 5
        assert all(isinstance(resp, str) for resp in data["responses"])

    def test_multiple_nos_endpoint_with_count(self):
        response = client.get("/nos?count=3")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert len(data["responses"]) == 3

    def test_multiple_nos_endpoint_count_limit(self):
        response = client.get("/nos?count=25")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 20  # round down to a max of 20
        assert len(data["responses"]) == 20

    def test_multiple_nos_endpoint_minimum_count(self):
        response = client.get("/nos?count=0")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["responses"]) == 1

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "markov_trained" in data
        assert "training_phrases_count" in data
        assert data["status"] == "healthy"
        assert data["markov_trained"] is True
        assert data["training_phrases_count"] == len(TRAINING_PHRASES)


class TestIntegration:
    def test_api_generates_different_responses(self):
        responses = []
        # test unique responses
        for _ in range(10):
            response = client.get("/no")
            data = response.json()
            responses.append(data["response"])
        unique_responses = set(responses)
        assert len(unique_responses) > 1, "Should generate varied responses"

    def test_training_data_is_valid(self):
        assert len(TRAINING_PHRASES) > 0
        for phrase in TRAINING_PHRASES:
            assert isinstance(phrase, str)
            assert len(phrase.strip()) > 0
            assert phrase.strip() != ""

    def test_api_responses_are_reasonable_length(self):
        for _ in range(5):
            response = client.get("/no")
            data = response.json()
            response_text = data["response"]
            word_count = len(response_text.split())
            assert 1 <= word_count <= 20, f"Response too long/short: {
                response_text}"

    def test_generation_time_header(self):
        response = client.get("/no")
        assert response.status_code == 200
        assert "X-Generation-Time-Ms" in response.headers
        time_str = response.headers["X-Generation-Time-Ms"]
        time_float = float(time_str)
        assert time_float >= 0, "Generation time should be non-negative"
        assert time_float < 1000, "Generation time should be reasonable (< 1 second)"

    def test_multiple_generation_time_header(self):
        response = client.get("/nos?count=3")
        assert response.status_code == 200
        assert "X-Generation-Time-Ms" in response.headers
        time_str = response.headers["X-Generation-Time-Ms"]
        time_float = float(time_str)
        assert time_float >= 0, "Generation time should be non-negative"
        assert time_float < 1000, "Generation time should be reasonable (< 1 second)"
