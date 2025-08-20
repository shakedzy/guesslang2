"""Machine learning model"""

import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from copy import deepcopy
from operator import itemgetter


LOGGER = logging.getLogger(__name__)

# Define modes without using deprecated ModeKeys
class ModeKeys:
    TRAIN = 'train'
    EVAL = 'valid' 
    PREDICT = 'test'

DATASET = {
    ModeKeys.TRAIN: 'train',
    ModeKeys.EVAL: 'valid',
    ModeKeys.PREDICT: 'test',
}


class HyperParameter:
    """Model hyper parameters"""
    BATCH_SIZE = 100
    NB_TOKENS = 10000
    VOCABULARY_SIZE = 5000
    EMBEDDING_SIZE = max(10, int(VOCABULARY_SIZE**0.5))
    DNN_HIDDEN_UNITS = [512, 32]
    DNN_DROPOUT = 0.5
    N_GRAM = 2


class Training:
    """Model training parameters"""
    SHUFFLE_BUFFER = HyperParameter.BATCH_SIZE * 10
    CHECKPOINT_STEPS = 1000
    LONG_TRAINING_STEPS = 10 * CHECKPOINT_STEPS
    SHORT_DELAY = 60
    LONG_DELAY = 5 * SHORT_DELAY


def load(saved_model_dir: str) -> tf.keras.Model:
    """Load a Keras model"""
    try:
        # Try loading as a Keras model first
        return tf.keras.models.load_model(saved_model_dir)
    except Exception:
        # Fallback to saved_model format for compatibility
        return tf.saved_model.load(saved_model_dir)


def build(model_dir: str, labels: list[str]) -> tf.keras.Model:
    """Build a Keras text classifier """
    
    # Input layer for preprocessed text features (n-grams as strings)
    text_input = tf.keras.Input(shape=(HyperParameter.NB_TOKENS,), dtype=tf.string, name='content')
    
    # Create string lookup layer to convert n-grams to integers
    string_lookup = tf.keras.utils.StringLookup(
        vocabulary=None, 
        mask_token="", 
        num_oov_indices=1,
        max_tokens=HyperParameter.VOCABULARY_SIZE,
        output_mode='int'
    )
    
    # Hash the input strings to integers
    hashed_input = string_lookup(text_input)
    
    # Embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=HyperParameter.VOCABULARY_SIZE,
        output_dim=HyperParameter.EMBEDDING_SIZE,
        mask_zero=True
    )(hashed_input)
    
    # Global average pooling to reduce sequence dimension
    pooled_embedding = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer)
    
    # Linear part (simple dense layer without activation)
    linear_output = tf.keras.layers.Dense(len(labels), name='linear_part')(pooled_embedding)
    
    # DNN part
    dnn_hidden = pooled_embedding
    for units in HyperParameter.DNN_HIDDEN_UNITS:
        dnn_hidden = tf.keras.layers.Dense(units, activation='relu')(dnn_hidden)
        dnn_hidden = tf.keras.layers.Dropout(HyperParameter.DNN_DROPOUT)(dnn_hidden)
    
    dnn_output = tf.keras.layers.Dense(len(labels), name='dnn_part')(dnn_hidden)
    
    # Combine linear and DNN outputs
    combined_output = tf.keras.layers.Add()([linear_output, dnn_output])
    
    # Final output with softmax activation
    predictions = tf.keras.layers.Softmax(name='predictions')(combined_output)
    
    # Create the model
    model = tf.keras.Model(inputs=text_input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Store labels for later use
    model.label_vocabulary = labels
    
    return model


def train(model: tf.keras.Model, data_root_dir: str, max_steps: int) -> dict[str, float]:
    """Train a Keras model"""
    
    LOGGER.debug('Building datasets')
    train_dataset = _build_keras_dataset(data_root_dir, ModeKeys.TRAIN, model.label_vocabulary)
    eval_dataset = _build_keras_dataset(data_root_dir, ModeKeys.EVAL, model.label_vocabulary)
    
    # Calculate epochs based on max_steps and dataset size
    # Estimate dataset size (this is approximate)
    approx_train_size = 1000  # Default estimate
    try:
        # Try to get actual size if possible
        approx_train_size = sum(1 for _ in train_dataset.take(1000)) * 10  # rough estimate
    except:
        pass
    
    steps_per_epoch = max(1, approx_train_size // HyperParameter.BATCH_SIZE)
    epochs = max(1, max_steps // steps_per_epoch)
    
    LOGGER.debug(f'Training for {epochs} epochs with ~{steps_per_epoch} steps per epoch')
    
    # Set up callbacks
    callbacks = []
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=eval_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Return training metrics
    final_metrics = {}
    if history.history:
        final_metrics['accuracy'] = history.history['accuracy'][-1]
        final_metrics['loss'] = history.history['loss'][-1]
        if 'val_accuracy' in history.history:
            final_metrics['val_accuracy'] = history.history['val_accuracy'][-1]
        if 'val_loss' in history.history:
            final_metrics['val_loss'] = history.history['val_loss'][-1]
    
    return final_metrics


def save(model: tf.keras.Model, saved_model_dir: str) -> None:
    """Save a Keras model"""
    Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
    model.save(saved_model_dir)
    
    # Save label vocabulary as well for compatibility
    if hasattr(model, 'label_vocabulary'):
        labels_file = Path(saved_model_dir) / 'labels.json'
        with open(labels_file, 'w') as f:
            json.dump(model.label_vocabulary, f)


def test(
    model: tf.keras.Model,
    data_root_dir: str,
    mapping: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Test a Keras model"""
    values = {language: 0 for language in mapping.values()}
    matches = {language: deepcopy(values) for language in values}

    LOGGER.debug('Test the model')
    
    # Load labels if available
    labels = getattr(model, 'label_vocabulary', list(mapping.keys()))
    
    test_dataset = _build_keras_dataset(data_root_dir, ModeKeys.PREDICT, labels, batch_size=1)
    
    for batch in test_dataset:
        content_batch, label_batch = batch
        
        # Get predictions
        predictions = model.predict(content_batch, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_label = labels[predicted_idx]
        
        # Get true label
        true_idx = int(label_batch.numpy()[0])
        true_label = labels[true_idx]
        
        # Map to languages
        label_language = mapping[true_label]
        predicted_language = mapping[predicted_label]
        matches[label_language][predicted_language] += 1

    return matches


def predict(
    model,  # Can be tf.keras.Model or AutoTrackable
    mapping: dict[str, str],
    text: str
) -> list[tuple[str, float]]:
    """Infer a model (Keras or legacy SavedModel)"""
    
    # Check if this is a Keras model or legacy SavedModel
    if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
        # Modern Keras model
        processed_text = _preprocess_text(tf.constant([text]))
        processed_text = tf.expand_dims(processed_text, 0)  # Add batch dimension
        
        predictions = model.predict(processed_text, verbose=0)
        probabilities = predictions[0]  # Remove batch dimension
        
        # Load labels if available
        labels = getattr(model, 'label_vocabulary', list(mapping.keys()))
        
    else:
        # Legacy SavedModel format - use signatures
        content_tensor = tf.constant([text])
        
        # Try different signature names
        signature_name = 'serving_default'
        if hasattr(model, 'signatures'):
            if signature_name in model.signatures:
                predicted = model.signatures[signature_name](content_tensor)
            elif 'predict' in model.signatures:
                predicted = model.signatures['predict'](content_tensor)
            else:
                # Take the first available signature
                signature_name = list(model.signatures.keys())[0]
                predicted = model.signatures[signature_name](content_tensor)
            
            # Extract probabilities and classes
            if 'scores' in predicted:
                probabilities = predicted['scores'][0].numpy()
            elif 'probabilities' in predicted:
                probabilities = predicted['probabilities'][0].numpy()
            else:
                # Fallback - look for any tensor that looks like probabilities
                prob_key = [k for k in predicted.keys() if 'prob' in k.lower() or 'score' in k.lower()]
                if prob_key:
                    probabilities = predicted[prob_key[0]][0].numpy()
                else:
                    # Use the first output tensor
                    probabilities = list(predicted.values())[0][0].numpy()
            
            # Get class labels
            if 'classes' in predicted:
                extensions = predicted['classes'][0].numpy()
                labels = [ext.decode() for ext in extensions]
            else:
                # Use mapping keys as fallback
                labels = list(mapping.keys())
        else:
            raise ValueError("Model has no signatures - cannot predict")
    
    # Create language-probability pairs
    language_scores = []
    for i, prob in enumerate(probabilities):
        if i < len(labels):
            extension = labels[i]
            language = mapping.get(extension, extension)  # fallback to extension if not in mapping
            language_scores.append((language, float(prob)))
    
    # Sort by probability (descending)
    scores = sorted(language_scores, key=itemgetter(1), reverse=True)
    return scores


def _build_keras_dataset(
    data_root_dir: str,
    mode: str,
    labels: list[str],
    batch_size: int = None
) -> tf.data.Dataset:
    """Build a tf.data.Dataset for Keras training"""
    if batch_size is None:
        batch_size = HyperParameter.BATCH_SIZE if mode != ModeKeys.PREDICT else 1
    
    pattern = str(Path(data_root_dir).joinpath(DATASET[mode], '*'))
    
    dataset = tf.data.Dataset.list_files(pattern, shuffle=(mode == ModeKeys.TRAIN))
    dataset = dataset.map(_read_file, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Create label mapping
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    def map_labels(content, label_string):
        # Convert string label to integer index
        label_str = tf.strings.as_string(label_string)
        # Find the index of the label
        label_idx = tf.py_function(
            lambda x: label_to_idx.get(x.numpy().decode(), 0),
            [label_str],
            tf.int32
        )
        label_idx.set_shape(())
        return content, label_idx
    
    dataset = dataset.map(map_labels, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(_preprocess_for_keras, num_parallel_calls=tf.data.AUTOTUNE)
    
    if mode == ModeKeys.TRAIN:
        dataset = dataset.shuffle(Training.SHUFFLE_BUFFER)
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def _preprocess_for_keras(
    data: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Process input data for Keras training"""
    processed_data = _preprocess_text(data)
    return processed_data, label


def _read_file(filename: str) -> tuple[tf.Tensor, tf.Tensor]:
    """Read a source file, return the content and the extension"""
    data = tf.io.read_file(filename)
    label = tf.strings.split([filename], '.').values[-1]
    return data, label


def _preprocess_text(data: tf.Tensor) -> tf.Tensor:
    """Feature engineering"""
    # Handle both single string and batch of strings
    original_shape = tf.shape(data)
    is_scalar = tf.equal(tf.rank(data), 0)
    
    # Convert scalar to 1D tensor if needed
    data = tf.cond(is_scalar, lambda: tf.expand_dims(data, 0), lambda: data)
    
    # Apply preprocessing to each element in the batch
    def preprocess_single(text):
        # Convert text to bytes and create n-grams
        bytes_split = tf.strings.bytes_split(text)
        ngrams = tf.strings.ngrams(bytes_split, HyperParameter.N_GRAM)
        
        # Pad or truncate to fixed size
        padding = tf.constant([''] * HyperParameter.NB_TOKENS, dtype=tf.string)
        padded = tf.concat([ngrams, padding], axis=0)
        truncated = padded[:HyperParameter.NB_TOKENS]
        return truncated
    
    # Process each element in the batch
    processed = tf.map_fn(preprocess_single, data, dtype=tf.string, parallel_iterations=32)
    
    # If input was scalar, return single processed element
    return tf.cond(is_scalar, lambda: processed[0], lambda: processed)
