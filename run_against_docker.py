import sys
import os
sys.path.insert(0, 'src')

os.environ.setdefault('LLM_PROVIDER', 'openai')
os.environ.setdefault('LLM_MODEL', 'llama-3.1-8b-instant')

from envs.silicon_flow_ppa.client import SiliconFlowPPAClient
from envs.silicon_flow_ppa.inference import llm_act

base_url = os.environ.get("PPA_SERVER_URL", "http://localhost:8000")
client = SiliconFlowPPAClient(base_url=base_url)
result = client.reset()

while not result.done:
    action = llm_act(result.observation)
    result = client.step(action)
    print(f"Placed {action.block_id} at ({action.x:.3f},{action.y:.3f}) reward={result.reward:.3f}")

print(client.render())