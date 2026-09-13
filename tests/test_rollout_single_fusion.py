import importlib.util
from pathlib import Path
import unittest
import json
from io import BytesIO
import tempfile
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('rollout', Path(__file__).resolve().parents[1] / 'scripts/rollout_single_fusion.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RolloutTest(unittest.TestCase):
    def test_remove_only_discovery_labels_from_retiring_service(self):
        labels = {'caddy': 'http://old', 'caddy_1.reverse_proxy': 'upstream', 'com.docker.stack.namespace': 'qwen-tts', 'owner': 'team'}
        self.assertEqual(module.no_caddy(labels), {'com.docker.stack.namespace': 'qwen-tts', 'owner': 'team'})

    def test_concurrent_environment_edit_blocks_update(self):
        before = {'id': 202, 'name': 'qwen-tts', 'endpoint': 4, 'env': [{'name': 'A', 'value': 'old'}], 'text': 'services: {}'}
        after = dict(before, env=[{'name': 'A', 'value': 'new'}])
        with patch.object(module, 'stack_snapshot', return_value=after), patch.object(module, 'api') as api:
            with self.assertRaises(AssertionError):
                module.update_stack(before, 'services: {}')
            api.assert_not_called()

    def test_service_lookup_rejects_missing_or_ambiguous_identity(self):
        for values in [[], [{'Spec': {'Name': 'expected'}}, {'Spec': {'Name': 'expected'}}]]:
            with patch.object(module, 'services', return_value=values):
                with self.assertRaises(AssertionError):
                    module.by_name('expected')

    def test_candidate_smoke_cannot_authorize_production_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'evidence.json').write_text(json.dumps({'status':'success','gradio':True,'phase':'candidate','source_commit':'commit','image':'image','target_url':'http://qwen-fusion-candidate','started_at':10,'completed_at':20}))
            with self.assertRaises(AssertionError):
                module.verify_evidence(path,{'commit':'commit','image':'image','created_at':1,'promoted_at':30},'production')

    def test_mixed_caddy_upstreams_block_retirement(self):
        config={'apps':{'http':{'servers':{'s':{'routes':[
            {'match':[{'host':[host]}],'handle':[{'handler':'reverse_proxy','upstreams':[{'dial':'10.0.3.1:8000'},{'dial':'10.0.3.2:8000'}]}]} for host in module.HOSTS
        ]}}}}}
        with patch.object(module,'by_name',return_value={'ID':'service-id','Endpoint':{'VirtualIPs':[{'Addr':'10.0.3.1/24'}]}}), patch.object(module,'api',return_value=[]), patch.object(module.urllib.request,'urlopen',return_value=BytesIO(json.dumps(config).encode())):
            with self.assertRaises(AssertionError):
                module.verify_routes()

    def test_caddy_may_route_directly_to_this_services_running_task(self):
        config={'apps':{'http':{'servers':{'s':{'routes':[
            {'match':[{'host':[host]}],'handle':[{'handler':'reverse_proxy','upstreams':[{'dial':'10.0.3.2:8000'}]}]} for host in module.HOSTS
        ]}}}}}
        task={'ServiceID':'service-id','Status':{'State':'running'},'NetworksAttachments':[{'Addresses':['10.0.3.2/24']}]}
        with patch.object(module,'by_name',return_value={'ID':'service-id','Endpoint':{'VirtualIPs':[{'Addr':'10.0.3.1/24'}]}}), patch.object(module,'api',return_value=[task]), patch.object(module.urllib.request,'urlopen',return_value=BytesIO(json.dumps(config).encode())):
            module.verify_routes()


if __name__ == '__main__':
    unittest.main()
