import sys
import os
sys.path.insert(0, 'src')

os.environ['LLM_PROVIDER'] = 'openai'
os.environ['LLM_MODEL'] = 'llama-3.1-8b-instant'
os.environ['LLM_API_KEY'] = os.environ.get('LLM_API_KEY', '') # paste your actual gsk_... key
os.environ['LLM_BASE_URL'] = 'https://api.groq.com/openai/v1/'

from envs.silicon_flow_ppa.client import SiliconFlowPPAClient
from envs.silicon_flow_ppa.inference import llm_act

client = SiliconFlowPPAClient(base_url="http://localhost:8000")
result = client.reset()

while not result.done:
    action = llm_act(result.observation)
    result = client.step(action)
    print(f"Placed {action.block_id} at ({action.x:.3f},{action.y:.3f}) reward={result.reward:.3f}")

print(client.render())