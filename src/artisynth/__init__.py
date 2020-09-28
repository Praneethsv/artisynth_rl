from gym.envs.registration import register
from common import constants as c

register(
    id='SpineEnv-v0',
    entry_point='artisynth.envs:SpineEnv',
    nondeterministic=False
)

register(
    id='Point2PointEnv-v5',
    entry_point='artisynth.envs:PointModelEnv',
    nondeterministic=False,
    kwargs={'ip': 'localhost',
            'port': 6030,
            'wait_action': '0.05'
            }
)