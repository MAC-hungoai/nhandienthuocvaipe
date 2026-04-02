import json

h = json.load(open('checkpoints/history.json'))

# Check if balanced_acc in history
if 'val_balanced_acc' in h:
    bal_accs = h['val_balanced_acc']
    print("🎯 Val Balanced Accuracy by Epoch:")
    for i, ba in enumerate(bal_accs, 1):
        print(f"  Epoch {i}: {ba:.4f}")
    
    # Find best
    best_ba = max(bal_accs)
    best_ba_epoch = bal_accs.index(best_ba) + 1
    print(f"\n✅ Best balanced_acc: {best_ba:.4f} at epoch {best_ba_epoch}")
    print(f"⚠️ Last balanced_acc: {bal_accs[-1]:.4f}")
    
    # Check improvement pattern
    print(f"\n📊 Improvement analysis:")
    for i in range(1, len(bal_accs)):
        improved = bal_accs[i] > bal_accs[i-1]
        diff = bal_accs[i] - bal_accs[i-1]
        print(f"  Epoch {i}→{i+1}: {'+' if improved else ''}{diff:+.4f} {'✅' if improved else '❌'}")
        
else:
    print("❌ val_balanced_acc not in history!")
    print(f"History keys: {list(h.keys())}")
