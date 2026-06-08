"""
Improved training script with better strategy.
"""
import os
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from data_loader import load_data
from model_builder_v2 import build_model, unfreeze_model, get_preprocess_fn


def train_model(data_dir='Skin_Conditions', img_size=(224, 224), 
                initial_epochs=25, fine_tune_epochs=30, base_model='ResNet50'):
    """
    Train with improved strategy.
    """
    print("=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)
    
    preprocess_fn = get_preprocess_fn(base_model)
    train_gen, val_gen, test_gen, class_names = load_data(
        data_dir=data_dir, 
        img_size=img_size,
        preprocess_fn=preprocess_fn
    )
    # Compute class weights to improve minority-class recall (e.g., Vitiligo)
    y_train = train_gen.classes
    unique_classes = np.unique(y_train)
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    class_weights = {int(c): float(w) for c, w in zip(unique_classes, class_weights_arr)}
    print("\nClass weights:", class_weights)

    
    # Save class names
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    print("\n" + "=" * 60)
    print(f"Building model with {base_model}...")
    print("=" * 60)
    
    model, base_model = build_model(
        num_classes=len(class_names),
        img_size=img_size,
        base_model_name=base_model
    )
    
    print(f"\nModel summary:")
    model.summary()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Phase 1: Train with frozen base
    print("\n" + "=" * 60)
    print("Phase 1: Training with frozen base model...")
    print("=" * 60)
    
    callbacks_phase1 = [
        ModelCheckpoint(
            'models/best_model_phase1.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger('logs/training_phase1.csv')
    ]
    
    history1 = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning (unfreezing base model)...")
    print("=" * 60)
    
    model = unfreeze_model(base_model, model, learning_rate=0.0001)
    
    callbacks_phase2 = [
        ModelCheckpoint(
            'models/best_model_finetuned.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=6,
            min_lr=1e-8,
            verbose=1
        ),
        CSVLogger('logs/training_phase2.csv')
    ]
    
    history2 = model.fit(
        train_gen,
        epochs=fine_tune_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")

    # Detailed per-class metrics to track confusion (Melasma vs Eczema, Vitiligo recall)
    y_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    cm = confusion_matrix(y_true, y_pred)
    print("\nClassification Report:\n")
    print(report)
    print("Confusion Matrix:\n", cm)
    
    # Save final model
    model.save('models/final_model.h5')
    print("\nModel saved to 'models/final_model.h5'")
    
    # Save history
    import pickle
    history = {
        'phase1': history1.history,
        'phase2': history2.history,
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss)
    }
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    with open('models/classification_report.txt', 'w') as f:
        f.write(report)
    np.save('models/confusion_matrix.npy', cm)
    
    print("\nTraining completed successfully!")
    return model, history


if __name__ == '__main__':
    # Try ResNet50 first (more reliable)
    train_model(base_model='ResNet50')

