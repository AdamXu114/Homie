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


# 修改方向
1. 同一奖励双阶段训练的reward scales可以变化
2. 上肢策略在下肢训练时可以运行，但是不更新command
3. compute reward 修改 done
4. reward scales 修改
5. 左右手采样具体逻辑修改 done
6. 采样固定位置的lpy和rpy范围 已获得,需要询问师兄确认 done