import json

h = json.load(open('checkpoints/history.json'))
epochs = len(h['train_loss'])

print(f"✅ Epochs trained: {epochs}/50")
print(f"📉 Last val_loss: {h['val_loss'][-1]:.4f}")
print(f"🎯 Best val_loss: {min(h['val_loss']):.4f} at epoch {h['val_loss'].index(min(h['val_loss']))+1}")
print(f"\n⚠️ CONVERGENCE ANALYSIS:")
print(f"Val loss trend (last 3): {[f'{v:.4f}' for v in h['val_loss'][-3:]]}")
print(f"Val acc trend (last 3): {[f'{v:.2%}' for v in h['val_acc'][-3:]]}")

# Check overfitting
best_epoch = h['val_loss'].index(min(h['val_loss'])) + 1
current_val_loss = h['val_loss'][-1]
best_val_loss = min(h['val_loss'])

if current_val_loss > best_val_loss:
    diff = current_val_loss - best_val_loss
    print(f"\n❌ OVERFITTING DETECTED:")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Val loss degradation: {diff:.4f} ({diff/best_val_loss*100:.1f}% increase)")
    print(f"   Model needs earlier stopping OR different regularization")
