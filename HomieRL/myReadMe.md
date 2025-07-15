### Train
```
python legged_gym/legged_gym/scripts/train.py --task g1 --num_envs 2048 --headless --resume
```

### Play
```
python legged_gym/legged_gym/scripts/play.py --num_envs 32 --task g1 --resume
```

### Export Policy
```
python legged_gym/legged_gym/scripts/export_onnx.py
```

### Mujoco Deploy
```
python legged_gym/deploy/deploy_mujoco/mujoco_deploy_copy.py g1.yaml
```
