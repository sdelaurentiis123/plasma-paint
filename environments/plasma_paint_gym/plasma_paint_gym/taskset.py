"""Native Verifiers v1 interaction wrapper; evaluation only until reward audit."""
import json
from pathlib import Path
from urllib.parse import urlparse
from typing import Literal
import verifiers.v1 as vf
from verifiers.v1.utils.image import image_data_url
from plasma_painter.config import sha256_file
from plasma_painter.training.painting_gym import PaintingGym


class PlasmaPaintGymData(vf.TaskData):
    frame: dict


class PlasmaPaintGymTask(vf.Task[PlasmaPaintGymData]):
    async def validate(self, runtime):
        PaintingGym(self.data.frame)


class PlasmaPaintGymConfig(vf.TasksetConfig):
    section: Literal[0,18,31] = 18
    cache_root: str = 'artifacts/plasma_painter/free_sections'


class PlasmaPaintGymTaskset(vf.Taskset[PlasmaPaintGymTask, PlasmaPaintGymConfig]):
    def load(self):
        root=Path(self.config.cache_root)
        index=json.loads((root/'index.json').read_text())
        record=next(r for r in index['records'] if r['y']==self.config.section)
        path=root/f'section-y{self.config.section}.generated.json'
        if sha256_file(path)!=record['sha256']:raise ValueError('Feature hash mismatch')
        clip=json.loads(path.read_text())
        if clip['split']!='art_train':raise ValueError('Only art_train is allowed')
        tasks=[]
        for i,frame in enumerate(clip['frames']):
            PaintingGym(frame)
            tasks.append(PlasmaPaintGymTask(PlasmaPaintGymData(idx=i,frame=frame,prompt=None),self.config.task))
        return tasks


def require_local_agent(agent):
    # Verifiers defaults to hosted inference. Never inherit that endpoint silently.
    url=agent.ctx.client.base_url
    if urlparse(url).scheme!='http' or urlparse(url).hostname not in {'127.0.0.1','localhost','::1'}:
        raise ValueError('Private plasma observations require an explicitly configured loopback inference endpoint')
    agent.trainable=False


class PlasmaPaintGymEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        require_local_agent(agents.agent)
        gym=PaintingGym(task.data.frame)
        feedback=None
        async with agents.agent.interaction(task) as interaction:
            while not gym.done:
                obs=gym.observe()
                parts=[vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=image_data_url(obs[k])))
                       for k in ('scientific','canvas')]
                parts.append(vf.TextContentPart(text='Paint the first image onto the second (current canvas). '
                    'Return one JSON action: paint with strokes, undo, or finish. All coordinates are normalized '
                    'XY in [0,1], not pixels. No JavaScript. Tools: '+json.dumps(obs['tools'])+
                    f' Turns left: {obs["turns_left"]}. Feedback: '+json.dumps(feedback)))
                segment=await interaction.turn([vf.UserMessage(content=parts)])
                if segment.terminated:break
                try:action=json.loads(segment.last_reply)
                except ValueError:action={'action':'invalid_json'}
                *_,feedback=gym.step(action)
            trace=interaction.trace
            trace.info['painting_gym']={'events':gym.events,'operations':gym.operations,
                'scientific_training_eligible':False,'trained_policy':False}
            trace.record_metric('accepted_action_fraction',sum(e['accepted'] for e in gym.events)/max(1,len(gym.events)))
            trace.record_metric('coarse_correspondence_diagnostic',gym.observe()['coarse_correspondence'])
            # No record_reward: this partial metric is not a scientific RL reward.
