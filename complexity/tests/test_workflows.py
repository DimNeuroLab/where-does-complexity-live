"""Checks for explicit RT-table routing and stage-specific inputs."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from complexity import run
from complexity.evaluation.compare_paper_tables import compare


class WorkflowTests(unittest.TestCase):
  def invoke(self, root: Path, profile: str, stages: list[str] | None = None, **config):
    filename = root / 'config.json'
    filename.write_text(json.dumps({'output_dir': str(root / 'output'), **config}))
    argv = ['complexity.run', '--config', str(filename), '--profile', profile]
    if stages:
      argv += ['--stages', *stages]
    with patch.object(sys, 'argv', argv), patch.object(run, 'execute') as execute:
      run.main()
    return execute.call_args_list

  def test_rt_tables_dispatch_only_the_unfiltered_suite(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      calls = self.invoke(root, 'coco-rt-tables', inputs={
        'coco_scanpaths': str(root / 'scanpaths.json'), 'human_scanpaths': '${UNUSED_HUMAN}',
        'nsd_metadata': '${UNUSED_NSD}',
      })
      self.assertEqual([call.args[0] for call in calls], [
        'complexity.preprocessing.convert_scanpaths', 'complexity.models.fit_response_time',
        'complexity.exports.export_rankings', 'complexity.evaluation.compare_paper_tables',
      ])
      self.assertNotIn('--only_correct', calls[0].args[1])
      self.assertIn(str(root / 'output/trials/coco_unfiltered.csv'), calls[0].args[1])
      self.assertIn(str(root / 'output/coco_unfiltered_rt'), calls[1].args[1])
      self.assertEqual(calls[-1].args[1][2:5], ['--tables', '1', 'A1'])

  def test_existing_all_profile_preserves_human_and_successful_inputs(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      calls = self.invoke(root, 'all', ['prepare'], inputs={
        key: str(root / key) for key in ['coco_scanpaths', 'human_scanpaths', 'nsd_scanpaths', 'nsd_metadata']
      })
      conversions = [call.args[1] for call in calls if call.args[0].endswith('.convert_scanpaths')]
      self.assertEqual(len(conversions), 3)
      self.assertNotIn('--only_correct', conversions[0])
      for argv in conversions[1:]:
        self.assertEqual(argv[-2:], ['--only_correct', '1'])
      self.assertTrue(any(call.args[0].endswith('.select_coco_trials') for call in calls))
      self.assertFalse(any('coco_unfiltered' in str(call) for call in calls))

  def test_postprocessing_needs_no_raw_data_configuration(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      calls = self.invoke(Path(directory), 'coco-rt-tables', ['compare'])
      self.assertEqual(len(calls), 1)
      self.assertTrue(calls[0].args[0].endswith('.compare_paper_tables'))

  def test_invalid_rt_stage_or_missing_input_fails_before_output_creation(self) -> None:
    for stages, inputs in [(['variance'], {}), (['prepare'], {}),
                           (['prepare'], {'coco_scanpaths': '${MISSING_SCANPATHS}'})]:
      with self.subTest(stages=stages, inputs=inputs), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with self.assertRaises(SystemExit) as error:
          self.invoke(root, 'coco-rt-tables', stages, inputs=inputs)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((root / 'output').exists())

  def test_generation_uses_coco_configuration_for_rt_tables(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      calls = self.invoke(root, 'coco-rt-tables', ['generate', 'prepare'],
        inputs={'coco_images': str(root / 'images'), 'coco_bboxes': str(root / 'boxes.json')},
        generation={'checkout': str(root / 'scandiff'), 'python': sys.executable,
                    'profiles': {'coco': {'precision': 'float32', 'devices': ['cuda:0']}}})
      self.assertIn('float32', calls[0].args[1])
      self.assertIn(str(root / 'output/scanpaths/coco/all_scanpaths.json'), calls[1].args[1])
      self.assertNotIn('--only_correct', calls[1].args[1])

  def test_rt_preparation_keeps_failed_and_unknown_target_present_trials(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      base = {'subject': 1, 'name': 'image.png', 'task': 'bottle', 'condition': 'present',
              'X': [0, 1], 'Y': [0, 1], 'T': [100, 100], 'RT': 200, 'length': 2, 'bbox': [0, 0, 2, 2]}
      source = root / 'scanpaths.json'
      source.write_text(json.dumps([dict(base, correct=flag) for flag in [0, 1, None]]
                                   + [dict(base, condition='absent', correct=1)]))
      config = root / 'config.json'
      config.write_text(json.dumps({'output_dir': str(root / 'output'), 'inputs': {'coco_scanpaths': str(source)}}))
      result = subprocess.run([sys.executable, '-m', 'complexity.run', '--config', str(config),
                               '--profile', 'coco-rt-tables', '--stages', 'prepare'], capture_output=True, text=True)
      self.assertEqual(result.returncode, 0, result.stderr)
      with (root / 'output/trials/coco_unfiltered.csv').open() as stream:
        rows = list(csv.DictReader(stream))
      self.assertEqual(len(rows), 3)
      self.assertEqual([row['correct'] for row in rows], ['0', '1', ''])

  def test_paper_reference_routes_rt_tables_to_unfiltered_suite(self) -> None:
    reference = Path(run.__file__).parent / 'reference/paper_tables.csv'
    with tempfile.TemporaryDirectory() as directory:
      rows = compare(Path(directory), reference, ['1', 'A1'])
      self.assertEqual(len(rows), 117)
      self.assertEqual({row['suite'] for row in rows}, {'coco_unfiltered_rt'})
      self.assertEqual({row['status'] for row in rows}, {'missing'})
      with self.assertRaisesRegex(ValueError, 'Unknown paper tables'):
        compare(Path(directory), reference, ['unknown'])


if __name__ == '__main__':
  unittest.main()
