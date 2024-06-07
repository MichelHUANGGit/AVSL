
# Train new model, validate on train, infer gallery to queries
python3 code/main.py --epochs 2 --batch_size_training 30 --batch_size_inference 30 --accumulation_steps 1 --lr 0.0001 --base_model_name ResNet50 --lay_to_emb_ids 1 2 3 4 --emb_dim 90 --num_classes 30 --device cuda --topk 32 --momentum 0.5 --momentum_decay 0.95 --p 2 --CNN_coeffs 32.0 0.1 --sim_coeffs 32.0 0.1 --train_model --infer_gallery_to_queries --validate_on_val --name test_AVSL --metrics_K 1 2 4 8
