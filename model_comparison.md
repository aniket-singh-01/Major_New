# Model Comparison Guide

Your training pipeline has produced three different model files:

## 1. best_model.h5
- This is the model saved by the `ModelCheckpoint` callback during training
- It represents the best performing model on the validation set
- It is saved whenever the monitored metric (usually `val_accuracy`) improves
- This model has not undergone any feature optimization

## 2. optimized_model.h5
- This is the model produced by the Grey Wolf Optimizer
- It contains only the most important features identified by the GWO
- It should be more efficient and potentially more interpretable
- It may have slightly different performance characteristics than the original model

## 3. dermatological_diagnosis_model.h5
- This appears to be the final model saved at the end of your training pipeline
- It likely incorporates all the improvements from your training process
- It represents the complete workflow including feature selection

## Which Model Should You Use?

### For best overall performance:
- Try **dermatological_diagnosis_model.h5** first as it represents your complete pipeline

### For fastest inference:
- Try **optimized_model.h5** as it should have fewer features and potentially faster inference

### For baseline comparison:
- Try **best_model.h5** as a reference point to see how much improvement the other models provide

## Recommendation

1. Start with **dermatological_diagnosis_model.h5** for most use cases
2. If you need faster predictions or resource-constrained deployment, use **optimized_model.h5**
3. You can also run a quick comparison using the test_model.py script to evaluate all three
