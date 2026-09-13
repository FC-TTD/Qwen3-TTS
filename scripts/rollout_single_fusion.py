#!/usr/bin/env python3
"""Stage and promote one managed Qwen fusion from a committed stack definition."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request
import urllib.parse
import yaml

NAME = 'qwen-tts-fusion-single'
OLD_NAME = 'qwen-tts-fusion'
BASE_ID = 202
ENDPOINT = 4
HOSTS = ['huolieniao', 'qwen-fusion', 'qianwen-api-design', 'qianwen-api-custom', 'qwen-api']


def api(path, method='GET', data=None):
    req = urllib.request.Request(os.environ.get('PT_URL', 'http://ttd-cctv:9000').rstrip('/') + path,
        method=method, headers={'X-API-Key': os.environ['PT_API_KEY'], 'Content-Type':'application/json'},
        data=None if data is None else json.dumps(data).encode())
    with urllib.request.urlopen(req, timeout=180) as response:
        content=response.read()
        return json.loads(content) if content else None


def snapshot(path, value):
    path.write_text(json.dumps(value, indent=2)); path.chmod(0o600)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def services():
    return api(f'/api/endpoints/{ENDPOINT}/docker/services')


def by_name(name):
    values=[s for s in services() if s['Spec']['Name']==name]
    assert len(values)==1, f'Expected one service {name}, found {len(values)}'
    return values[0]


def stack_snapshot(sid):
    meta=api(f'/api/stacks/{sid}')
    return {'id':sid,'name':meta['Name'],'endpoint':meta['EndpointId'], 'env':meta.get('Env',[]),
            'text':api(f'/api/stacks/{sid}/file')['StackFileContent']}


def update_stack(before,text):
    assert signature(stack_snapshot(before['id']))==signature(before), 'Portainer management changed concurrently'
    api(f"/api/stacks/{before['id']}?endpointId={ENDPOINT}", 'PUT',
        {'stackFileContent':text,'env':before['env'],'prune':True,'pullImage':False})
    after=stack_snapshot(before['id'])
    assert yaml.safe_load(after['text'])==yaml.safe_load(text)
    return after


def no_caddy(labels):
    return {k:v for k,v in labels.items() if not k.startswith('caddy')}


def stage_candidate(args,root):
    repo=Path(__file__).resolve().parents[1]
    subprocess.run(['git','diff','--exit-code',args.commit,'--','docker/stack.fusion.yml'],cwd=repo,check=True)
    assert args.image.startswith('registry.ttd/qwen-tts-fusion/fusion@sha256:') and len(args.image.rsplit(':',1)[1])==64
    assert not any(s['Spec']['Name'].startswith(NAME+'_') for s in services()), 'Candidate/final already exists; inspect before resuming'
    assert not any(s['Name']==NAME for s in api('/api/stacks')), 'Managed name already exists'
    text=subprocess.check_output(['git','show',f'{args.commit}:docker/stack.fusion.yml'],cwd=repo,text=True)
    desired=yaml.safe_load(text);assert set(desired['services'])=={'fusion'}
    model=desired['services']['fusion'];model['image']=args.image
    assert model['deploy']['replicas']==1
    assert model['deploy']['placement']['constraints']==['node.hostname == ttd-worker']
    assert 'NVIDIA_VISIBLE_DEVICES=2' in model['environment']
    model['environment'] += [f'APP_GIT_COMMIT={args.commit}',f'APP_IMAGE={args.image}']
    candidate=copy.deepcopy(desired)
    candidate['services']['fusion']['deploy']['labels']={
        'caddy':'http://qwen-fusion-candidate','caddy.reverse_proxy':'{{upstreams 8000}}',
        'caddy.reverse_proxy.health_uri':'/health','caddy.reverse_proxy.health_interval':'10s',
        'caddy.reverse_proxy.health_timeout':'5s'}
    base=stack_snapshot(BASE_ID);assert base['name']=='qwen-tts' and base['endpoint']==ENDPOINT
    old=by_name(OLD_NAME+'_fusion')
    expected_base={'qwen-tts_base-1','qwen-tts_base-2'}
    assert {s['Spec']['Name'] for s in services() if s['Spec']['Name'].startswith('qwen-tts_')}==expected_base
    assert not any(s['Name']==OLD_NAME for s in api('/api/stacks')), 'Old fusion management changed; re-audit'
    info=api(f'/api/endpoints/{ENDPOINT}/docker/info');swarm=info['Swarm']['Cluster']['ID']
    record={'commit':args.commit,'image':args.image,'swarm_id':swarm,'desired':desired,'base':base,'old':old,'phase':'prepared','created_at':time.time()}
    snapshot(root/'state.json',record)
    created=api(f'/api/stacks/create/swarm/string?endpointId={ENDPOINT}','POST',
        {'Name':NAME,'SwarmID':swarm,'StackFileContent':yaml.safe_dump(candidate,sort_keys=False),'Env':[],'FromAppTemplate':False})
    assert created['Name']==NAME and created['EndpointId']==ENDPOINT
    record.update({'new_id':created['Id'],'candidate':stack_snapshot(created['Id']),'phase':'candidate_created'})
    snapshot(root/'state.json',record)
    print(f"Candidate stack {created['Id']} created; validate http://qwen-fusion-candidate before promotion",flush=True)


def verify_evidence(path,record,phase):
    evidence=json.loads((path/'evidence.json').read_text())
    assert evidence['status']=='success' and evidence['gradio'] is True
    assert evidence['source_commit']==record['commit'] and evidence['image']==record['image']
    assert evidence['phase']==phase
    expected_url='http://qwen-fusion-candidate' if phase=='candidate' else 'http://qwen-fusion'
    assert evidence['target_url']==expected_url
    assert evidence['completed_at']>=evidence['started_at']>=record['created_at' if phase=='candidate' else 'promoted_at']
    records=evidence['records']
    assert {r['name'] for r in records}>={'design','custom','clone','clone-slow','clone-fast','clone-aligned'}
    for name in ['design','custom','clone','ui-design','ui-custom','ui-clone']:
        assert (path/(name+'.wav')).stat().st_size>1000


def backend(host,record):
    with urllib.request.urlopen('http://'+host+'/health/backends',timeout=15) as response:
        data=json.load(response)
    assert data['local']['self_contained'] is True and data['local']['ready'] is True
    assert data['deployment']=={'source_commit':record['commit'],'image':record['image']}


def verify_routes():
    service=by_name(NAME+'_fusion')
    addresses={item['Addr'].split('/')[0]+':8000' for item in service.get('Endpoint',{}).get('VirtualIPs',[])}
    addresses.update({NAME+'_fusion:8000',NAME+'_fusion.:8000'})
    filters=urllib.parse.quote(json.dumps({'service':[service['ID']],'desired-state':['running']}))
    tasks=api(f'/api/endpoints/{ENDPOINT}/docker/tasks?filters={filters}')
    for task in tasks:
        if task.get('ServiceID')==service['ID'] and task.get('Status',{}).get('State')=='running':
            for network in task.get('NetworksAttachments',[]):
                addresses.update(address.split('/')[0]+':8000' for address in network.get('Addresses',[]))
    with urllib.request.urlopen('http://ttd-server/caddy_api/config/',timeout=15) as response:
        config=json.load(response)
    found={name:set() for name in HOSTS}
    def walk(routes,hosts=()):
        for route in routes:
            for match in route.get('match') or [{}]:
                selected=match.get('host',hosts)
                for handler in route.get('handle',[]):
                    if handler.get('handler')=='subroute':walk(handler.get('routes',[]),selected)
                    if handler.get('handler')=='reverse_proxy':
                        for host in selected:
                            if host in found:found[host].update(x['dial'] for x in handler.get('upstreams',[]))
    for server in config['apps']['http']['servers'].values():walk(server.get('routes',[]))
    assert all(value and value<=addresses for value in found.values()), 'Caddy still has missing or mixed Qwen upstreams'


def promote(args,root,record):
    assert record['phase']=='candidate_created'
    verify_evidence(Path(args.evidence),record,'candidate')
    backend('qwen-fusion-candidate',record)
    assert signature(by_name(OLD_NAME+'_fusion')['Spec'])==signature(record['old']['Spec']), 'Old fusion spec changed concurrently'
    assert signature(stack_snapshot(BASE_ID))==signature(record['base']), 'Old Base management changed concurrently'
    record['candidate']=update_stack(record['candidate'],yaml.safe_dump(record['desired'],sort_keys=False))
    record['phase']='new_production_labels';snapshot(root/'state.json',record)
    old=by_name(OLD_NAME+'_fusion')
    assert signature(old['Spec'])==signature(record['old']['Spec'])
    spec=copy.deepcopy(old['Spec']);spec['Labels']=no_caddy(spec.get('Labels',{}))
    api(f"/api/endpoints/{ENDPOINT}/docker/services/{old['ID']}/update?version={old['Version']['Index']}",'POST',spec)
    record['old_unrouted']=by_name(OLD_NAME+'_fusion')
    assert record['old_unrouted']['Spec']['TaskTemplate']==old['Spec']['TaskTemplate']
    record['phase']='old_fusion_unrouted';snapshot(root/'state.json',record)
    base=yaml.safe_load(record['base']['text'])
    for service in base['services'].values():
        service['deploy']['labels']=no_caddy(service['deploy'].get('labels',{}))
    record['base_unrouted']=update_stack(record['base'],yaml.safe_dump(base,sort_keys=False))
    record['phase']='old_base_unrouted';snapshot(root/'state.json',record)
    finish_promotion(root,record)


def finish_promotion(root,record):
    assert record['phase']=='old_base_unrouted'
    assert signature(stack_snapshot(record['new_id']))==signature(record['candidate'])
    assert signature(stack_snapshot(BASE_ID))==signature(record['base_unrouted'])
    assert signature(by_name(OLD_NAME+'_fusion')['Spec'])==signature(record['old_unrouted']['Spec'])
    for attempt in range(30):
        try:
            verify_routes()
            for host in HOSTS:backend(host,record)
            break
        except Exception:
            if attempt==29:raise
            time.sleep(3)
    record.update({'phase':'promoted','promoted_at':time.time()});snapshot(root/'state.json',record)
    print('Five production aliases now reach the self-contained fusion; run final real smoke before retirement',flush=True)


def drained(name):
    command=['docker','ps','-q','--filter',f'label=com.docker.swarm.service.name={name}']
    container=subprocess.check_output(['ssh','root@ttd-worker',shlex.join(command)],text=True).strip()
    if not container:return
    assert '\n' not in container
    probe="""from pathlib import Path
import urllib.request
urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=60).read()
n=0
for p in ('/proc/net/tcp','/proc/net/tcp6'):
 for line in Path(p).read_text().splitlines()[1:]:
  f=line.split();n+=f[1].split(':')[1]=='1F40' and f[3]=='01'
print(n)
"""
    for _ in range(30):
        r=subprocess.run(['ssh','root@ttd-worker',shlex.join(['docker','exec',container,'python','-c',probe])],text=True,capture_output=True,timeout=70)
        if r.returncode==0 and r.stdout.strip()=='0':return
        time.sleep(3)
    raise RuntimeError(f'{name} still has open requests; retained')


def retire(args,root,record):
    assert record['phase']=='promoted'
    verify_evidence(Path(args.evidence),record,'production')
    verify_routes()
    for host in HOSTS:backend(host,record)
    assert signature(stack_snapshot(record['new_id']))==signature(record['candidate'])
    assert signature(by_name(OLD_NAME+'_fusion')['Spec'])==signature(record['old_unrouted']['Spec'])
    assert signature(stack_snapshot(BASE_ID))==signature(record['base_unrouted'])
    for name in [OLD_NAME+'_fusion','qwen-tts_base-1','qwen-tts_base-2']:drained(name)
    verify_routes()
    assert signature(by_name(OLD_NAME+'_fusion')['Spec'])==signature(record['old_unrouted']['Spec'])
    assert {s['Spec']['Name'] for s in services() if s['Spec'].get('Labels',{}).get('com.docker.stack.namespace')==OLD_NAME}=={OLD_NAME+'_fusion'}
    api(f'/api/stacks/{OLD_NAME}?external=true&endpointId={ENDPOINT}','DELETE')
    record['phase']='old_fusion_removed';snapshot(root/'state.json',record)
    verify_routes()
    assert signature(stack_snapshot(BASE_ID))==signature(record['base_unrouted'])
    assert {s['Spec']['Name'] for s in services() if s['Spec'].get('Labels',{}).get('com.docker.stack.namespace')=='qwen-tts'}=={'qwen-tts_base-1','qwen-tts_base-2'}
    api(f'/api/stacks/{BASE_ID}?endpointId={ENDPOINT}','DELETE')
    record['phase']='old_base_removed';snapshot(root/'state.json',record)
    expected={NAME+'_fusion'}
    for attempt in range(30):
        names={s['Spec']['Name'] for s in services() if s['Spec']['Name'].startswith('qwen-tts')}
        if names==expected:break
        if attempt==29:raise RuntimeError(f'Unexpected remaining Qwen services: {names}')
        time.sleep(2)
    verify_routes()
    for host in HOSTS:backend(host,record)
    record.update({'phase':'complete','status':'success'});snapshot(root/'state.json',record)
    print('Only one managed Qwen fusion remains; all five aliases verified',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['candidate','promote','verify-promoted','retire']);p.add_argument('--state',required=True);p.add_argument('--commit');p.add_argument('--image');p.add_argument('--evidence');args=p.parse_args()
    root=Path(args.state);root.mkdir(parents=True,exist_ok=True);root.chmod(0o700)
    if args.phase=='candidate':stage_candidate(args,root)
    else:
        record=json.loads((root/'state.json').read_text())
        try:
            if args.phase=='verify-promoted':finish_promotion(root,record)
            else:(promote if args.phase=='promote' else retire)(args,root,record)
        except Exception as error:
            record['failure_type']=type(error).__name__;snapshot(root/'state.json',record);raise

if __name__=='__main__':main()
