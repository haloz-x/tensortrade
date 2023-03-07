# Copyright 2020 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import uuid
import logging

from typing import Dict, Any, Tuple
from random import randint

import gym
import numpy as np

from tensortrade.core import TimeIndexed, Clock, Component
from tensortrade.env.generic import (
    ActionScheme,
    RewardScheme,
    Observer,
    Stopper,
    Informer,
    Renderer
)


class TradingEnv(gym.Env, TimeIndexed):
    """A trading environment made for use with Gym-compatible reinforcement
    learning algorithms.

    Parameters
    ----------
    action_scheme : `ActionScheme`
        A component for generating an action to perform at each step of the
        environment.
    reward_scheme : `RewardScheme`
        A component for computing reward after each step of the environment.
    observer : `Observer`
        A component for generating observations after each step of the
        environment.
    informer : `Informer`
        A component for providing information after each step of the
        environment.
    renderer : `Renderer`
        A component for rendering the environment.
    kwargs : keyword arguments
        Additional keyword arguments needed to create the environment.
    """

    agent_id: str = None
    episode_id: str = None

    def __init__(self,
                 action_scheme: ActionScheme,
                 reward_scheme: RewardScheme,
                 observer: Observer,
                 stopper: Stopper,
                 informer: Informer,
                 renderer: Renderer,
                 min_periods: int = None,
                 max_episode_steps: int = None,
                 random_start_pct: float = 0.00,
                 **kwargs) -> None:
        super().__init__()
        self.clock = Clock()

        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.observer = observer
        self.stopper = stopper
        self.informer = informer
        self.renderer = renderer
        self.min_periods = min_periods
        self.random_start_pct = random_start_pct

        # Register the environment in Gym and fetch spec
        # 23/3/6 添加entry_point, 还不清除entry_point的作用
        gym.envs.register(
            id='TensorTrade-v0',
            entry_point='gym.envs.classic_control:TensorTrade',
            max_episode_steps=max_episode_steps,
        )
        self.spec = gym.spec(env_id='TensorTrade-v0')

        # 同步组件时钟
        for c in self.components.values():
            c.clock = self.clock

        self.action_space = action_scheme.action_space
        self.observation_space = observer.observation_space

        self._enable_logger = kwargs.get('enable_logger', False)
        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

    @property
    def components(self) -> 'Dict[str, Component]':
        """The components of the environment. (`Dict[str,Component]`, read-only)
            返回所有环境组件
        """
        return {
            "action_scheme": self.action_scheme,
            "reward_scheme": self.reward_scheme,
            "observer": self.observer,
            "stopper": self.stopper,
            "informer": self.informer,
            "renderer": self.renderer
        }

    def step(self, action: Any) -> 'Tuple[np.array, float, bool, dict]':
        """Makes one step through the environment.
        在环境中执行一步
        Parameters
        ----------
        action : Any
            An action to perform on the environment.

        Returns
        -------
        `np.array`
            The observation of the environment after the action being
            performed.
        float
            The computed reward for performing the action.
        bool
            Whether or not the episode is complete.
        dict
            The information gathered after completing the step.
        """
        # 对环境发起动作，买或卖
        self.action_scheme.perform(self, action)

        # 获得观察数据
        obs = self.observer.observe(self)

        # 奖励
        reward = self.reward_scheme.reward(self)

        done = self.stopper.stop(self)
        info = self.informer.info(self)

        #时钟计数
        self.clock.increment()

        return obs, reward, done, info

    def reset(self) -> 'np.array':
        """Resets the environment.
        环境数据从新开始
        Returns
        -------
        obs : `np.array`
            The first observation of the environment.
        """
        if self.random_start_pct > 0.00:
            size = len(self.observer.feed.process[-1].inputs[0].iterable)
            random_start = randint(0, int(size * self.random_start_pct))
        else:
            random_start = 0

        self.episode_id = str(uuid.uuid4())
        self.clock.reset()

        # 关联的每个组件调用reset方法， 重置状态
        for c in self.components.values():
            if hasattr(c, "reset"):
                if isinstance(c, Observer):
                    c.reset(random_start=random_start)
                else:
                    c.reset()

        # 重置后，获得预加载环境数据(目的是让代理有足够的观察数据)
        obs = self.observer.observe(self)

        self.clock.increment()

        return obs

    def render(self, **kwargs) -> None:
        """Renders the environment."""
        self.renderer.render(self, **kwargs)

    def save(self) -> None:
        """Saves the rendered view of the environment."""
        self.renderer.save()

    def close(self) -> None:
        """Closes the environment."""
        self.renderer.close()
