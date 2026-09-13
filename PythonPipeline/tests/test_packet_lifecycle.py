"""Topology/state regression cases, not fabricated cardistry observations."""
import unittest

from cardcap.cards.lifecycle import LifecycleConfig, PacketLifecycle


def evidence(supported=True):
    return {"kind": "synthetic_state_machine_test", "description": "Algebraic membership transitions; not a real video result",
            "supported_rigid_membership": supported}


class LifecycleTests(unittest.TestCase):
    def test_split_merge_conservation_and_nonoverlapping_lifetimes(self):
        engine = PacketLifecycle(LifecycleConfig(confirmation_frames=2))
        groups = [[['a', 'b']]] * 2 + [[['a'], ['b']]] * 2 + [[['a', 'b']]] * 2
        for frame, partition in enumerate(groups):
            engine.observe(frame, partition, evidence=evidence())
        data = engine.finish()
        self.assertEqual([n['birth_frame'] for n in data['packet_nodes']], [0, 2, 2, 4])
        self.assertEqual([n['death_frame'] for n in data['packet_nodes']], [1, 3, 3, 5])
        self.assertEqual(data['events']['splits'][0]['confirmed_at_frame'], 3)
        self.assertEqual(data['events']['merges'][0]['frame'], 4)
        for frame in range(6):
            members = [m for n in data['packet_nodes'] if n['birth_frame'] <= frame <= n['death_frame'] for m in n['membership_atoms']]
            self.assertEqual(sorted(members), ['a', 'b'])

    def test_unknown_does_not_merge_and_resets_confirmation(self):
        engine = PacketLifecycle(LifecycleConfig(confirmation_frames=2))
        timeline = [[['a'], ['b']], [['a'], ['b']], [['a', 'b']], None, [['a', 'b']]]
        for frame, groups in enumerate(timeline):
            engine.observe(frame, groups, evidence=evidence(groups is not None))
        data = engine.finish()
        self.assertEqual(data['events'], {'splits': [], 'merges': []})
        self.assertEqual(len(data['packet_nodes']), 2)
        self.assertTrue(all(n['death_frame'] == 4 for n in data['packet_nodes']))

    def test_transition_after_occlusion_preserves_observation_bracket(self):
        engine = PacketLifecycle(LifecycleConfig(confirmation_frames=2))
        timeline = [[['a'], ['b']], [['a'], ['b']], None, None, [['a', 'b']], [['a', 'b']]]
        for frame, groups in enumerate(timeline):
            engine.observe(frame, groups, evidence=evidence(groups is not None))
        event = engine.finish()['events']['merges'][0]
        self.assertEqual(event['frame'], 4)
        self.assertEqual(event['confirmed_at_frame'], 5)
        bracket = event['transition_observation_bracket']
        self.assertEqual(bracket['last_old_partition_observed_frame'], 1)
        self.assertEqual(bracket['possible_first_new_state_frame_interval'], [2, 4])
        self.assertFalse(bracket['unobserved_intermediate_topologies_excluded'])

    def test_return_to_old_partition_resets_transition_bracket(self):
        engine = PacketLifecycle(LifecycleConfig(confirmation_frames=2))
        timeline = [[['a', 'b']], [['a', 'b']], None, [['a'], ['b']], [['a', 'b']], [['a'], ['b']], [['a'], ['b']]]
        for frame, groups in enumerate(timeline):
            engine.observe(frame, groups, evidence=evidence(groups is not None))
        event = engine.finish()['events']['splits'][0]
        self.assertEqual(event['transition_observation_bracket']['possible_first_new_state_frame_interval'], [5, 5])
        self.assertEqual(event['confirmed_at_frame'], 6)

    def test_missing_member_and_duplicates_rejected_without_mutation(self):
        engine = PacketLifecycle(LifecycleConfig(confirmation_frames=1))
        engine.observe(0, [['a'], ['b']], evidence=evidence())
        for groups in ([['a']], [['a', 'a'], ['b']]):
            with self.assertRaises(ValueError):
                engine.observe(1, groups, evidence=evidence())
        engine.observe(1, None, evidence=evidence(False))
        self.assertEqual(len(engine.finish()['packet_nodes']), 2)

    def test_many_to_many_not_fabricated_as_split_then_merge(self):
        engine = PacketLifecycle(LifecycleConfig(confirmation_frames=1))
        engine.observe(0, [['a', 'b'], ['c', 'd']], evidence=evidence())
        engine.observe(1, [['a', 'c'], ['b', 'd']], evidence=evidence())
        data = engine.finish()
        self.assertEqual(len(data['packet_nodes']), 2)
        self.assertFalse(data['events']['splits'] or data['events']['merges'])
        self.assertEqual(data['diagnostics'][-1]['state'], 'unresolved_many_to_many_repartition')

    def test_all_unknown_creates_no_physical_packets(self):
        engine = PacketLifecycle()
        for frame in range(4):
            engine.observe(frame, None, evidence=evidence(False))
        data = engine.finish()
        self.assertEqual(data['packet_nodes'], [])
        self.assertEqual(len(data['diagnostics']), 4)
        self.assertEqual(engine.finish(), data)
        with self.assertRaises(RuntimeError):
            engine.observe(4, None, evidence=evidence(False))

    def test_order_missing_evidence_and_nonconsecutive_frames_rejected(self):
        engine = PacketLifecycle(LifecycleConfig(confirmation_frames=1))
        with self.assertRaises(ValueError):
            engine.observe(0, [['a']], evidence=evidence(False))
        engine.observe(0, [['a']], evidence=evidence())
        with self.assertRaises(ValueError):
            engine.observe(2, [['a']], evidence=evidence())
        engine.observe(1, [['a']], evidence=evidence())
        self.assertEqual(engine.finish()['packet_nodes'][0]['death_frame'], 1)


if __name__ == '__main__':
    unittest.main()
