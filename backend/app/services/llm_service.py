"""Service LLM pour l'intégration vLLM."""
from openai import OpenAI
from app.config import VLLM_HOST, VLLM_PORT

# Client vLLM global
vllm_client = None


def initialize_vllm_client():
    """Initialiser le client vLLM."""
    global vllm_client
    vllm_client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
    )
    print(f"Connected to vLLM at {VLLM_HOST}:{VLLM_PORT}")
    return vllm_client


def get_vllm_client():
    """Obtenir l'instance du client vLLM."""
    if vllm_client is None:
        raise RuntimeError("vLLM client not initialized")
    return vllm_client


def generate_completion(prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
    """Générer une complétion en utilisant vLLM."""
    client = get_vllm_client()
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content
