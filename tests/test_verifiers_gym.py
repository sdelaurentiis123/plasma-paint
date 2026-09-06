"""Optional native-v1 wrapper smoke with a scripted local agent, not model inference."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
pytest.importorskip('verifiers.v1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environments/plasma_paint_gym'))
from plasma_paint_gym.taskset import PlasmaPaintGymTaskset,PlasmaPaintGymConfig,PlasmaPaintGymEnv,require_local_agent


def test_native_taskset_and_interaction():
    tasks=PlasmaPaintGymTaskset(PlasmaPaintGymConfig()).load()
    assert len(tasks)==8
    metrics={}
    class Interaction:
        trace=SimpleNamespace(info={},record_metric=lambda name,value:metrics.update({name:value}))
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def turn(self,messages):
            assert len(messages[0].content)==3
            return SimpleNamespace(terminated=False,last_reply=json.dumps({'action':'finish'}))
    agent=SimpleNamespace(ctx=SimpleNamespace(client=SimpleNamespace(base_url='http://127.0.0.1:8000/v1')),
                          interaction=lambda task:Interaction(),trainable=True)
    asyncio.run(PlasmaPaintGymEnv.run(None,tasks[0],SimpleNamespace(agent=agent)))
    assert not agent.trainable
    assert metrics['coarse_correspondence_diagnostic']==0
    assert not Interaction.trace.info['painting_gym']['scientific_training_eligible']


def test_hosted_endpoint_rejected_before_observation():
    agent=SimpleNamespace(ctx=SimpleNamespace(client=SimpleNamespace(base_url='https://api.pinference.ai/api/v1')))
    with pytest.raises(ValueError,match='loopback'):require_local_agent(agent)
