"""决策层验收测试：:mod:`decision.cleaning_policy`（V1.0 固定趟数策略）。

重点验证固定趟数表、补偿永远只跑一趟、大小写不敏感、配置校验，以及
"策略层不暴露任何压力/PWM/速度控制" 这一 V1.0 契约。
"""
import unittest

import pytest

from decision.cleaning_policy import LEVELS, CleaningPolicy
from tests.helpers import CLEANING_CFG
from utils.errors import ConfigError, TaskError

#: V1.0 固定趟数表（验收契约，不随源码漂移）。
EXPECTED_PASSES = {'LIGHT': 1, 'MEDIUM': 1, 'HEAVY': 2}


def make_policy(**overrides):
    cfg = dict(light_passes=1, medium_passes=1, heavy_passes=2)
    cfg.update(overrides)
    return CleaningPolicy(cfg)


class CleaningPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = CleaningPolicy(dict(CLEANING_CFG))

    # ---------------------------------------------------------------- pass table
    def test_v1_pass_table(self):
        for level, passes in EXPECTED_PASSES.items():
            self.assertEqual(self.policy.passes(level), passes, level)
        self.assertEqual(self.policy.base_passes('LIGHT'), 1)
        self.assertEqual(self.policy.base_passes('MEDIUM'), 1)
        self.assertEqual(self.policy.base_passes('HEAVY'), 2)

    def test_configured_pass_table_matches_v1_contract(self):
        self.assertEqual(dict(CLEANING_CFG)['light_passes'], 1)
        self.assertEqual(dict(CLEANING_CFG)['medium_passes'], 1)
        self.assertEqual(dict(CLEANING_CFG)['heavy_passes'], 2)

    def test_compensation_is_always_one_pass(self):
        for level in LEVELS + ('LIGHT', 'HEAVY', 'MEDIUM'):
            self.assertEqual(self.policy.passes(level, compensation=True), 1, level)
        self.assertEqual(self.policy.compensation_passes(), 1)

    def test_compensation_short_circuits_unknown_level(self):
        # 补偿分支在查表之前返回，因此不会因为未知等级报错。
        self.assertEqual(self.policy.passes('UNKNOWN', compensation=True), 1)

    def test_helpers(self):
        self.assertTrue(self.policy.allows_compensation)
        self.assertEqual(self.policy.describe(), EXPECTED_PASSES)
        self.assertEqual(set(self.policy.describe()), set(LEVELS))

    # ---------------------------------------------------------------- levels
    def test_level_lookup_is_case_insensitive(self):
        for spelling in ('light', 'Light', 'LIGHT', 'lIgHt'):
            self.assertEqual(self.policy.passes(spelling), 1)
        for spelling in ('heavy', 'Heavy', 'HEAVY'):
            self.assertEqual(self.policy.passes(spelling), 2)
        self.assertEqual(self.policy.passes('medium'), 1)

    def test_unknown_level_raises_task_error(self):
        for level in ('BLACK', '', 'extreme', None, 3):
            with pytest.raises(TaskError):
                self.policy.passes(level)

    # ---------------------------------------------------------------- config
    def test_invalid_pass_values_raise_config_error(self):
        for overrides in ({'light_passes': 0}, {'medium_passes': -1},
                          {'heavy_passes': 0}, {'light_passes': 1.5},
                          {'heavy_passes': 'two'}, {'medium_passes': True},
                          {'heavy_passes': None}):
            with pytest.raises(ConfigError):
                make_policy(**overrides)

    def test_missing_pass_key_raises_config_error(self):
        for key in ('light_passes', 'medium_passes', 'heavy_passes'):
            cfg = dict(light_passes=1, medium_passes=1, heavy_passes=2)
            del cfg[key]
            with pytest.raises(ConfigError):
                CleaningPolicy(cfg)

    def test_higher_pass_counts_are_accepted(self):
        policy = make_policy(heavy_passes=3)
        self.assertEqual(policy.passes('HEAVY'), 3)

    # ---------------------------------------------------------------- V1 scope
    def test_policy_exposes_no_pressure_or_pwm_control(self):
        for name in ('set_pressure', 'set_pwm', 'set_speed', 'set_motor_speed',
                     'pressure', 'pwm', 'speed'):
            self.assertFalse(hasattr(self.policy, name), name)

    def test_config_keys_never_mention_pressure_pwm_or_speed(self):
        for key in self.policy.cfg:
            lowered = str(key).lower()
            for forbidden in ('pressure', 'pwm', 'speed', 'duty'):
                self.assertNotIn(forbidden, lowered, key)

    def test_module_import_does_not_add_actuator_helpers(self):
        self.assertFalse(hasattr(CleaningPolicy, 'set_pressure'))
        self.assertFalse(hasattr(CleaningPolicy, 'set_pwm'))


if __name__ == '__main__':
    unittest.main()
