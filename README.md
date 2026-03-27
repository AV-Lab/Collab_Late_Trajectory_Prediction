# Collab_Late_Trajectory_Prediction
CoLTP framework - Collaborative Multi-Agent Trajectory Prediction with Late Fusion

### OPV2V dataset
#### preprocess
python -m preprocess.OPV2V /media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/OPV2V

#### create prediction data
python create_trajecotry_data.py OPV2V /media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/OPV2V/train/train_data.pkl train --global
python create_trajecotry_data.py OPV2V /media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/OPV2V/valid/valid_data.pkl valid --global
python create_trajecotry_data.py OPV2V /media/nadya/86bf701c-9a26-47cf-89c1-3a952cb40cc1/OPV2V/test/test_data.pkl test --global

#### train predictor 
python -m train_predictor.training.train train_predictor/generators/data/OPV2V_L10_H20_S3_F10 train_predictor/checkpoints
