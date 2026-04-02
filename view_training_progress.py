import json
import matplotlib.pyplot as plt

try:
    h = json.load(open('checkpoints/history.json'))
    epochs = len(h['train_loss'])
    
    print(f"\n{'='*60}")
    print(f"📊 TRAINING PROGRESS - {epochs} epochs completed")
    print(f"{'='*60}")
    
    # Print epoch-by-epoch results
    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Train Acc':<12} {'Val Acc':<12}")
    print("-" * 60)
    
    for i in range(epochs):
        epoch = i + 1
        tl = h['train_loss'][i]
        vl = h['val_loss'][i]
        ta = h['train_acc'][i]
        va = h['val_acc'][i]
        print(f"{epoch:<8} {tl:<12.4f} {vl:<12.4f} {ta:<12.2%} {va:<12.2%}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"📈 SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    best_val_loss_epoch = h['val_loss'].index(min(h['val_loss'])) + 1
    best_val_acc_epoch = h['val_acc'].index(max(h['val_acc'])) + 1
    
    print(f"\n✅ Best Val Loss: {min(h['val_loss']):.4f} at epoch {best_val_loss_epoch}")
    print(f"✅ Best Val Acc:  {max(h['val_acc']):.2%} at epoch {best_val_acc_epoch}")
    print(f"⚠️  Last Val Loss: {h['val_loss'][-1]:.4f} (degradation: {h['val_loss'][-1] - min(h['val_loss']):.4f})")
    print(f"⚠️  Last Val Acc:  {h['val_acc'][-1]:.2%} (degradation: {max(h['val_acc']) - h['val_acc'][-1]:.2%})")
    
    print(f"\n✅ Total training time: {sum(h['epoch_time_sec']):.1f}s ({sum(h['epoch_time_sec'])/60:.1f} min)")
    print(f"✅ Avg time per epoch: {sum(h['epoch_time_sec'])/epochs:.1f}s")
    
    # Check for overfitting
    final_gap = h['train_loss'][-1] - h['val_loss'][-1]
    if final_gap < -0.2:
        print(f"\n⚠️  OVERFITTING DETECTED: train_loss < val_loss")
    elif h['val_loss'][-1] > min(h['val_loss']) * 1.1:
        print(f"\n⚠️  POSSIBLE OVERFITTING: val_loss degraded {((h['val_loss'][-1] / min(h['val_loss']) - 1) * 100):.1f}%")
    else:
        print(f"\n✅ NO SIGNIFICANT OVERFITTING")
    
    print(f"\n{'='*60}\n")
    
except FileNotFoundError:
    print("❌ history.json not found!")
