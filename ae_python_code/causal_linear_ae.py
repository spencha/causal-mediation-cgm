#!/usr/bin/env python3
"""
Causal Linear-Friendly Autoencoder (CLAE) - TensorFlow 2.x Compatible
======================================================================
Learns representations φ that:
1. Control for pre-treatment confounding effectively
2. Are optimized for use in linear/GAM models
3. Satisfy causal inference assumptions
4. Maintain interpretability and stability
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Input, Dense, LSTM, Concatenate, GaussianNoise,
    Masking, RepeatVector, Dropout, BatchNormalization,
    LayerNormalization, TimeDistributed, Lambda, Reshape,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import KFold

# ================================================================
# Custom Layers for Causal Constraints (Fixed for TF 2.x)
# ================================================================

class LinearizabilityPenalty(Layer):
    """
    Encourages φ to have linear relationships with outcomes
    by penalizing deviation from best linear fit
    """
    def __init__(self, alpha=0.1, y_dim=36, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.y_dim = y_dim
    
    def build(self, input_shape):
        super().build(input_shape)
        # input_shape is a list: [phi_shape, y_shape]
        phi_dim = input_shape[0][-1]
        
        # Create a Dense layer for linear projection
        self.linear_proj = Dense(self.y_dim, use_bias=True, name='linear_proj_layer')
    
    def call(self, inputs):
        phi, y_true = inputs  # phi: (B, d), y_true: (B, k)
        
        # Project φ linearly to y-space
        y_pred_linear = self.linear_proj(phi)
        
        # Penalize non-linearity: force φ to be linearly decodable
        linear_loss = tf.reduce_mean(tf.square(y_true - y_pred_linear))
        
        self.add_loss(self.alpha * linear_loss)
        return phi
    
    def compute_output_shape(self, input_shape):
        # Return the shape of phi (first input)
        return input_shape[0]


class ConditionalIndependencePenalty(Layer):
    """
    Encourages φ to satisfy conditional independence assumptions
    P(A ⊥ M | φ) for valid mediation analysis
    """
    def __init__(self, lambda_ci=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ci = lambda_ci
    
    def build(self, input_shape):
        super().build(input_shape)
        # input_shape is [phi_shape, A_shape, M_shape]
        phi_dim = input_shape[0][-1]
        
        # Create Dense layers for residualization
        self.a_proj = Dense(1, use_bias=True, name='a_residual_proj')
        self.m_proj = Dense(1, use_bias=True, name='m_residual_proj')
    
    def call(self, inputs):
        phi, A, M = inputs
        
        # Predict A and M from φ
        A_pred = self.a_proj(phi)
        M_pred = self.m_proj(phi)
        
        # Compute residuals
        A_resid = A - A_pred
        M_resid = M - M_pred
        
        # Correlation between residuals (should be near zero)
        # Standardize residuals
        A_resid_std = (A_resid - tf.reduce_mean(A_resid)) / (tf.math.reduce_std(A_resid) + 1e-8)
        M_resid_std = (M_resid - tf.reduce_mean(M_resid)) / (tf.math.reduce_std(M_resid) + 1e-8)
        
        # Compute correlation
        corr = tf.reduce_mean(A_resid_std * M_resid_std)
        
        ci_loss = tf.square(corr)
        self.add_loss(self.lambda_ci * ci_loss)
        return phi
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]


class BalancingScorePenalty(Layer):
    """
    Ensures φ acts as a balancing score:
    Treated and control units should be comparable given φ
    """
    def __init__(self, gamma_balance=0.1, n_moments=2, **kwargs):
        super().__init__(**kwargs)
        self.gamma_balance = gamma_balance
        self.n_moments = n_moments
    
    def call(self, inputs):
        phi, A_binary = inputs
        
        A_bin = tf.cast(tf.reshape(A_binary, (-1, 1)), tf.float32)
        
        # Compute moments of φ for treated/control
        phi_treated = phi * A_bin
        phi_control = phi * (1 - A_bin)
        
        n_treat = tf.reduce_sum(A_bin) + 1e-6
        n_control = tf.reduce_sum(1 - A_bin) + 1e-6
        
        # First moment (mean)
        mean_t = tf.reduce_sum(phi_treated, axis=0) / n_treat
        mean_c = tf.reduce_sum(phi_control, axis=0) / n_control
        balance_loss = tf.reduce_sum(tf.square(mean_t - mean_c))
        
        if self.n_moments >= 2:
            # Second moment (variance)
            # Compute centered versions
            phi_centered_t = (phi - tf.expand_dims(mean_t, 0)) * A_bin
            phi_centered_c = (phi - tf.expand_dims(mean_c, 0)) * (1 - A_bin)
            
            var_t = tf.reduce_sum(tf.square(phi_centered_t), axis=0) / n_treat
            var_c = tf.reduce_sum(tf.square(phi_centered_c), axis=0) / n_control
            balance_loss += 0.5 * tf.reduce_sum(tf.square(var_t - var_c))
        
        self.add_loss(self.gamma_balance * balance_loss)
        return phi
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]


class StabilityRegularizer(Layer):
    """
    Encourages stable representations across bootstrap samples
    """
    def __init__(self, lambda_stab=0.01, **kwargs):
        super().__init__(**kwargs)
        self.lambda_stab = lambda_stab
    
    def call(self, inputs):
        phi = inputs
        
        # Create bootstrap indices
        batch_size = tf.shape(phi)[0]
        indices = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=batch_size,
            dtype=tf.int32
        )
        
        # Bootstrap sample
        phi_boot = tf.gather(phi, indices)
        
        # Encourage similarity in covariance structure
        cov_orig = tf.matmul(phi, phi, transpose_a=True) / tf.cast(batch_size, tf.float32)
        cov_boot = tf.matmul(phi_boot, phi_boot, transpose_a=True) / tf.cast(batch_size, tf.float32)
        
        stability_loss = tf.reduce_mean(tf.square(cov_orig - cov_boot))
        self.add_loss(self.lambda_stab * stability_loss)
        return phi
    
    def compute_output_shape(self, input_shape):
        return input_shape


# ================================================================
# Basis Expansion for Linear Models
# ================================================================

class LearnedBasisExpansion(Layer):
    """
    Learns basis functions that make relationships more linear
    Similar to kernel trick but learned end-to-end
    FIXED: Now preserves input dimension instead of changing it
    """
    def __init__(self, n_bases=20, basis_type='rbf', **kwargs):
        super().__init__(**kwargs)
        self.n_bases = n_bases
        self.basis_type = basis_type
    
    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        
        if self.basis_type == 'rbf':
            # RBF centers
            self.centers = self.add_weight(
                name='centers',
                shape=(self.n_bases, input_dim),
                initializer='glorot_uniform',
                trainable=True
            )
            self.log_gamma = self.add_weight(
                name='log_gamma',
                shape=(self.n_bases,),
                initializer='zeros',
                trainable=True
            )
            # Project back to original dimension
            self.output_projection = Dense(
                input_dim,
                activation=None,
                kernel_regularizer=l2(1e-5),
                name='basis_to_latent'
            )
        elif self.basis_type == 'polynomial':
            # Polynomial projection
            self.poly_weights = self.add_weight(
                name='poly_weights',
                shape=(input_dim, self.n_bases),
                initializer='glorot_uniform',
                trainable=True
            )
            # Project back to original dimension
            self.output_projection = Dense(
                input_dim,
                activation=None,
                kernel_regularizer=l2(1e-5),
                name='poly_to_latent'
            )
    
    def call(self, inputs):
        if self.basis_type == 'rbf':
            # Compute RBF features
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x*c
            x2 = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
            c2 = tf.reduce_sum(tf.square(self.centers), axis=-1)
            xc = tf.matmul(inputs, self.centers, transpose_b=True)
            distances = x2 + c2 - 2 * xc
            
            gamma = tf.exp(self.log_gamma)
            rbf_features = tf.exp(-gamma * distances)  # Shape: (batch, n_bases)
            
            # Project back to input dimension
            output = self.output_projection(rbf_features)  # Shape: (batch, input_dim)
            
            # Residual connection to preserve information
            return output + inputs
            
        elif self.basis_type == 'polynomial':
            # Simple polynomial features
            linear_proj = tf.matmul(inputs, self.poly_weights)
            poly_features = tf.concat([
                linear_proj,
                tf.square(linear_proj),
                tf.nn.sigmoid(linear_proj)  # Some non-linearity
            ], axis=-1)
            poly_features = poly_features[:, :self.n_bases]  # Ensure correct size
            
            # Project back to input dimension
            output = self.output_projection(poly_features)
            
            # Residual connection
            return output + inputs
        
        return inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape  # Output shape = input shape (preserves dimension)


# ================================================================
# Encoder Architectures
# ================================================================

def build_lstm_encoder(inp_ts, inp_meal, inp_subj, T, p, n_meals, n_subs,
                       latent_dim=16, l2_reg=1e-4, use_categorical_features=False):
    """
    LSTM-based encoder for temporal CGM data.

    Architecture:
    - Bidirectional LSTM captures sequential dependencies
    - Better for long-range temporal patterns

    Parameters:
    -----------
    use_categorical_features : bool
        If False (default), exclude meal_ohe and subj_ohe from encoder.
        These features strongly correlate with treatment (carb amount) and
        cause poor balance scores. Set to False for causal inference.
    """
    # Time series processing
    if use_categorical_features:
        # Include meal type and subject embeddings (NOT recommended for balance)
        meal_emb = Dense(4, activation="relu")(inp_meal)
        subj_emb = Dense(4, activation="relu")(inp_subj)
        meal_t   = RepeatVector(T)(meal_emb)
        subj_t   = RepeatVector(T)(subj_emb)
        x = Concatenate()([inp_ts, meal_t, subj_t])
    else:
        # Use only time series data (recommended for better balance)
        # Meal type and subject ID leak treatment information
        x = inp_ts

    x = GaussianNoise(0.05)(x)  # Less noise
    x = Masking(mask_value=0.0)(x)

    # Single LSTM pathway (simpler is better)
    x = LSTM(64, return_sequences=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)  # Less dropout

    # Pre-treatment summary (important for confounding)
    # Use all timesteps since input is now pre-meal only (truncated to 24 timesteps)
    pre_mean = Lambda(lambda y: tf.reduce_mean(y, axis=1))(inp_ts)
    pre_features = Dense(32, activation="relu", kernel_regularizer=l2(l2_reg))(pre_mean)

    # Combine temporal and pre-treatment
    combined = Concatenate()([x, pre_features])
    combined = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.2)(combined)

    # Core latent representation (directly to latent_dim)
    phi_core = Dense(latent_dim, activation=None,
                     kernel_regularizer=l2(l2_reg),
                     name="phi_core")(combined)

    return phi_core


def build_cnn_encoder(inp_ts, inp_meal, inp_subj, T, p, n_meals, n_subs,
                      latent_dim=16, l2_reg=1e-4, use_categorical_features=False):
    """
    CNN-based encoder for temporal CGM data.

    Architecture:
    - 1D convolutions along time axis
    - Captures local patterns (glucose spikes, trends)
    - Generally faster to train than LSTM

    Parameters:
    -----------
    use_categorical_features : bool
        If False (default), exclude meal_ohe and subj_ohe from encoder.
        These features strongly correlate with treatment (carb amount) and
        cause poor balance scores. Set to False for causal inference.
    """
    # Time series processing
    if use_categorical_features:
        # Include meal type and subject embeddings (NOT recommended for balance)
        meal_emb = Dense(4, activation="relu")(inp_meal)
        subj_emb = Dense(4, activation="relu")(inp_subj)
        meal_t   = RepeatVector(T)(meal_emb)
        subj_t   = RepeatVector(T)(subj_emb)
        x = Concatenate()([inp_ts, meal_t, subj_t])
    else:
        # Use only time series data (recommended for better balance)
        x = inp_ts

    x = GaussianNoise(0.05)(x)

    # CNN blocks
    # Block 1: Capture short-term patterns (5-15 min)
    x = Conv1D(32, kernel_size=3, padding="same", activation="relu",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, padding="same", activation="relu",
               kernel_regularizer=l2(l2_reg))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # Block 2: Capture medium-term patterns (15-45 min)
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu",
               kernel_regularizer=l2(l2_reg))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # Block 3: Capture longer patterns (45-120 min)
    x = Conv1D(128, kernel_size=3, padding="same", activation="relu",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)

    # Pre-treatment summary (same as LSTM version for fair comparison)
    # Use all timesteps since input is now pre-meal only (truncated to 24 timesteps)
    pre_mean = Lambda(lambda y: tf.reduce_mean(y, axis=1))(inp_ts)
    pre_features = Dense(32, activation="relu", kernel_regularizer=l2(l2_reg))(pre_mean)

    # Combine CNN features with pre-treatment summary
    combined = Concatenate()([x, pre_features])
    combined = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.2)(combined)

    # Latent representation
    phi_core = Dense(latent_dim, activation=None,
                     kernel_regularizer=l2(l2_reg),
                     name="phi_core")(combined)

    return phi_core


# ================================================================
# Main Causal Linear-Friendly Autoencoder
# ================================================================

def build_causal_linear_ae(
    T, p, n_meals, n_subs,
    latent_dim=16,
    n_basis_functions=20,
    l2_reg=1e-4,
    encoder_type="lstm",  # NEW: "lstm" or "cnn"
    use_linearization=True,
    use_balancing=True,
    use_ci_penalty=True,
    use_stability=True,
    lambda_linear=0.1,
    lambda_balance=0.1,
    lambda_ci=0.05,
    lambda_stab=0.01,
    use_categorical_features=False  # NEW: exclude meal/subj for better balance
):
    """
    Build autoencoder optimized for linear downstream models

    Parameters:
    -----------
    encoder_type : str
        "lstm" - Use LSTM-based encoder (default, captures sequential dependencies)
        "cnn" - Use CNN-based encoder (faster, captures local patterns)
    use_categorical_features : bool
        If False (default), exclude meal_ohe and subj_ohe from encoder.
        These features strongly correlate with treatment and cause poor balance.
    """
    # Inputs
    inp_ts    = Input((T, p),      name="ts_input")
    inp_meal  = Input((n_meals,),  name="meal_input")
    inp_subj  = Input((n_subs,),   name="subj_input")
    inp_A     = Input((1,),        name="treat_continuous")
    inp_A_bin = Input((1,),        name="treat_binary")
    inp_M     = Input((1,),        name="mediator_input")
    inp_Y     = Input((36,),       name="outcome_seq")

    # Select encoder architecture
    if encoder_type.lower() == "cnn":
        phi_core = build_cnn_encoder(
            inp_ts, inp_meal, inp_subj, T, p, n_meals, n_subs,
            latent_dim=latent_dim, l2_reg=l2_reg,
            use_categorical_features=use_categorical_features
        )
    else:
        # Default LSTM encoder
        phi_core = build_lstm_encoder(
            inp_ts, inp_meal, inp_subj, T, p, n_meals, n_subs,
            latent_dim=latent_dim, l2_reg=l2_reg,
            use_categorical_features=use_categorical_features
        )
    
    # Apply basis expansion for linear-friendliness (now preserves dimension)
    phi_basis = LearnedBasisExpansion(
        n_bases=n_basis_functions,
        basis_type='rbf',
        name="basis_expansion"
    )(phi_core)
    
    # Normalize
    phi_final = LayerNormalization(center=True, scale=True, name="phi")(phi_basis)
    
    # Apply causal penalties
    if use_linearization:
        phi_final = LinearizabilityPenalty(
            alpha=lambda_linear,
            y_dim=36,
            name="linearization"
        )([phi_final, inp_Y])
    
    if use_balancing:
        phi_final = BalancingScorePenalty(
            gamma_balance=lambda_balance,
            n_moments=2,
            name="balancing"
        )([phi_final, inp_A_bin])
    
    if use_ci_penalty:
        phi_final = ConditionalIndependencePenalty(
            lambda_ci=lambda_ci,
            name="cond_independence"
        )([phi_final, inp_A, inp_M])
    
    if use_stability:
        phi_final = StabilityRegularizer(
            lambda_stab=lambda_stab,
            name="stability"
        )(phi_final)
    
    # Reconstruction heads (to ensure φ captures relevant info)
    # 1. Predict pre-treatment covariates
    pre_recon = Dense(32, activation="relu")(phi_final)
    pre_recon = Dense(24 * p, activation=None, name="pre_recon")(pre_recon)
    
    # 2. Predict treatment propensity
    a_logit = Dense(16, activation="relu")(phi_final)
    a_logit = Dense(1, activation=None, name="a_logit")(a_logit)
    
    # 3. Predict mediator
    m_pred = Dense(16, activation="relu")(phi_final)
    m_pred = Dense(1, activation=None, name="m_pred")(m_pred)
    
    # 4. Predict outcome (simplified)
    y_pred = Dense(32, activation="relu")(phi_final)
    y_pred = Dense(36, activation=None, name="y_pred")(y_pred)
    
    model = Model(
        inputs=[inp_ts, inp_meal, inp_subj, inp_A, inp_A_bin, inp_M, inp_Y],
        outputs=[pre_recon, a_logit, m_pred, y_pred]
    )
    
    # Encoder model (for extracting φ)
    encoder = Model(
        inputs=[inp_ts, inp_meal, inp_subj],
        outputs=model.get_layer("phi").output
    )
    
    return model, encoder


# ================================================================
# Training Functions
# ================================================================

def train_causal_linear_ae(
    X_ts_pre, meal_ohe, subj_ohe,
    A_cont, M_scalar, Y_seq,
    latent_dim=16,
    n_basis_functions=20,
    epochs=50,
    batch_size=128,
    lr=1e-3,
    verbose=2,
    seed=123,
    validation_split=0.2,
    encoder_type="lstm",  # NEW: "lstm" or "cnn"
    optimizer_name="adamw",  # NEW: "adam", "adamw", "sgd", "rmsprop", "nadam"
    use_linearization=True,  # NEW: Exposed penalty flags
    use_balancing=True,
    use_ci_penalty=True,
    use_stability=True,
    lambda_linear=0.1,
    lambda_balance=2.0,  # INCREASED from 0.5 to further improve balance
    lambda_ci=0.05,
    lambda_stab=0.01,
    treatment_head_weight=0.0,  # NEW: Set to 0 to disable treatment prediction head
    treatment_median=None,  # NEW: Optional median for consistent binarization
    use_categorical_features=False  # NEW: Exclude meal/subj from encoder for better balance
):
    """
    Train the causal linear-friendly autoencoder with configurable optimizer and architecture.

    Parameters:
    -----------
    encoder_type : str
        "lstm" or "cnn" - encoder architecture to use
    optimizer_name : str
        "adam", "adamw", "sgd", "rmsprop", "nadam" - optimizer to use
    use_linearization : bool
        Whether to apply linearization penalty
    use_balancing : bool
        Whether to apply balancing score penalty
    use_ci_penalty : bool
        Whether to apply conditional independence penalty
    use_stability : bool
        Whether to apply stability regularizer
    treatment_head_weight : float
        Weight for treatment prediction head loss. Set to 0.0 to disable (recommended
        for balance, since predicting treatment conflicts with balance objective).
    treatment_median : float or None
        If provided, use this median for binarizing treatment. If None, compute from data.
    use_categorical_features : bool
        If False (default), exclude meal_ohe and subj_ohe from encoder input.
        These features correlate strongly with treatment and cause poor balance.
    """
    tf.keras.utils.set_random_seed(seed)

    n, T, p = X_ts_pre.shape
    n_meals = meal_ohe.shape[1]
    n_subs = subj_ohe.shape[1]

    # Prepare targets - use provided median or compute from data
    if treatment_median is not None:
        A_bin = (A_cont > treatment_median).astype(np.float32)
    else:
        A_bin = (A_cont > np.median(A_cont)).astype(np.float32)

    # Flatten pre-treatment for reconstruction target
    pre_target = X_ts_pre[:, :24, :].reshape(n, -1).astype(np.float32)

    # Standardize continuous variables
    A_std = ((A_cont - np.mean(A_cont)) / (np.std(A_cont) + 1e-8)).astype(np.float32)
    M_std = ((M_scalar - np.mean(M_scalar)) / (np.std(M_scalar) + 1e-8)).astype(np.float32)
    Y_std = ((Y_seq - np.mean(Y_seq)) / (np.std(Y_seq) + 1e-8)).astype(np.float32)

    # Build model
    model, encoder = build_causal_linear_ae(
        T=T, p=p, n_meals=n_meals, n_subs=n_subs,
        latent_dim=latent_dim,
        n_basis_functions=n_basis_functions,
        l2_reg=1e-4,
        encoder_type=encoder_type,
        use_linearization=use_linearization,
        use_balancing=use_balancing,
        use_ci_penalty=use_ci_penalty,
        use_stability=use_stability,
        lambda_linear=lambda_linear,
        lambda_balance=lambda_balance,
        lambda_ci=lambda_ci,
        lambda_stab=lambda_stab,
        use_categorical_features=use_categorical_features
    )

    # Optimizer selection
    optimizer_configs = {
        "adam": lambda: tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        "adamw": lambda: tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5, clipnorm=1.0),
        "sgd": lambda: tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipnorm=1.0),
        "rmsprop": lambda: tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0),
        "nadam": lambda: tf.keras.optimizers.Nadam(learning_rate=lr, clipnorm=1.0),
    }

    try:
        opt_fn = optimizer_configs.get(optimizer_name.lower(), optimizer_configs["adamw"])
        opt = opt_fn()
    except (AttributeError, TypeError):
        # Fallback for older TensorFlow versions
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    
    model.compile(
        optimizer=opt,
        loss={
            'pre_recon': 'mse',
            'a_logit': 'mse',  # Changed from binary_crossentropy for stability
            'm_pred': 'mse',
            'y_pred': 'mse'
        },
        loss_weights={
            'pre_recon': 0.5,   # Reduced
            'a_logit': treatment_head_weight,  # Default 0.0 - disabled to avoid conflict with balance
            'm_pred': 0.5,      # Keep moderate
            'y_pred': 2.0       # Increased - focus on outcomes
        }
    )
    
    # Prepare inputs - reshape scalars to have shape (n, 1)
    inputs = [
        X_ts_pre.astype(np.float32),
        meal_ohe.astype(np.float32),
        subj_ohe.astype(np.float32),
        A_std.reshape(-1, 1).astype(np.float32),      # Reshape to (n, 1)
        A_bin.reshape(-1, 1).astype(np.float32),      # Reshape to (n, 1)
        M_std.reshape(-1, 1).astype(np.float32),      # Reshape to (n, 1)
        Y_std.astype(np.float32)
    ]
    
    targets = [
        pre_target,
        A_bin.reshape(-1, 1).astype(np.float32),
        M_std.reshape(-1, 1).astype(np.float32),
        Y_std
    ]
    
    # Train
    history = model.fit(
        inputs, targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
        shuffle=True
    )
    
    # Extract features
    phi = encoder.predict([X_ts_pre, meal_ohe, subj_ohe], verbose=0)
    
    return model, encoder, phi, history


# ================================================================
# Validation Functions
# ================================================================

def validate_linear_friendliness(phi, Y, A, M, n_folds=5):
    """
    Validate that φ works well with linear models
    """
    results = {}
    
    # Test 1: Linear predictability of Y
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=123)
    
    # Simple linear model
    linear_scores = []
    for train_idx, test_idx in kf.split(phi):
        phi_train, phi_test = phi[train_idx], phi[test_idx]
        y_train, y_test = Y[train_idx, 0], Y[test_idx, 0]  # First outcome
        
        ridge = RidgeCV(alphas=np.logspace(-4, 4, 20))
        ridge.fit(phi_train, y_train)
        score = ridge.score(phi_test, y_test)
        linear_scores.append(score)
    
    results['linear_R2'] = np.mean(linear_scores)
    
    # Test 2: Check if polynomial expansion improves much
    poly_scores = []
    for train_idx, test_idx in kf.split(phi):
        phi_train, phi_test = phi[train_idx], phi[test_idx]
        y_train, y_test = Y[train_idx, 0], Y[test_idx, 0]
        
        # Add polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        phi_poly_train = poly.fit_transform(phi_train)
        phi_poly_test = poly.transform(phi_test)
        
        # Limit features to avoid overfitting
        max_features = min(phi_poly_train.shape[1], 100)
        phi_poly_train = phi_poly_train[:, :max_features]
        phi_poly_test = phi_poly_test[:, :max_features]
        
        ridge = RidgeCV(alphas=np.logspace(-4, 4, 20))
        ridge.fit(phi_poly_train, y_train)
        score = ridge.score(phi_poly_test, y_test)
        poly_scores.append(score)
    
    results['poly_R2'] = np.mean(poly_scores)
    results['linearity_ratio'] = results['linear_R2'] / (results['poly_R2'] + 1e-8)
    
    # Test 3: Balance check
    A_bin = (A > np.median(A)).astype(int)
    treated_mean = np.mean(phi[A_bin == 1], axis=0)
    control_mean = np.mean(phi[A_bin == 0], axis=0)
    std_diff = np.abs(treated_mean - control_mean) / (np.std(phi, axis=0) + 1e-8)
    results['max_std_diff'] = np.max(std_diff)
    results['mean_std_diff'] = np.mean(std_diff)
    
    return results


def validate_causal_assumptions(phi, A, M, Y, pre_X):
    """
    Validate causal assumptions are satisfied
    """
    results = {}
    
    # Test 1: Conditional independence A ⊥ M | φ
    ridge_a = RidgeCV(alphas=np.logspace(-4, 4, 20))
    ridge_m = RidgeCV(alphas=np.logspace(-4, 4, 20))
    
    ridge_a.fit(phi, A)
    ridge_m.fit(phi, M)
    
    resid_a = A - ridge_a.predict(phi)
    resid_m = M - ridge_m.predict(phi)
    
    corr = np.corrcoef(resid_a, resid_m)[0, 1]
    results['residual_correlation_AM'] = abs(corr)
    
    # Test 2: Overlap (propensity score)
    A_bin = (A > np.median(A)).astype(int)
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(phi, A_bin)
    ps = ps_model.predict_proba(phi)[:, 1]
    
    results['ps_min'] = np.min(ps)
    results['ps_max'] = np.max(ps)
    results['ps_overlap'] = np.min([np.max(ps[A_bin == 0]), np.max(ps[A_bin == 1])]) - \
                            np.max([np.min(ps[A_bin == 0]), np.min(ps[A_bin == 1])])
    
    # Test 3: Sufficient dimension reduction
    if pre_X is not None:
        pre_X_flat = pre_X.reshape(pre_X.shape[0], -1)
        ridge_pre = RidgeCV(alphas=np.logspace(-4, 4, 20))
        ridge_pre.fit(phi, pre_X_flat[:, 0])  # Just first feature as test
        results['pre_X_R2'] = ridge_pre.score(phi, pre_X_flat[:, 0])
    
    return results


# ================================================================
# Export for use with R
# ================================================================

def export_causal_features(phi, global_ids, meal_list, subj_list, 
                          A, M, Y, save_path):
    """
    Export φ features with metadata for R analysis
    """
    import pandas as pd
    
    # Create dataframe
    phi_df = pd.DataFrame(phi, columns=[f'phi_{i+1}' for i in range(phi.shape[1])])
    
    # Add metadata
    phi_df['global_window_id'] = global_ids
    phi_df['meal_type'] = meal_list
    phi_df['subject_id'] = subj_list
    phi_df['treat_meal_carbs'] = A
    phi_df['mediator_bolus_for_meal'] = M
    
    # Add validation metrics as attributes
    val_linear = validate_linear_friendliness(phi, Y, A, M)
    val_causal = validate_causal_assumptions(phi, A, M, Y, None)  # Need pre_X
    
    # Save with metadata
    phi_df.to_csv(save_path, index=False)
    
    # Save validation results
    val_path = save_path.replace('.csv', '_validation.json')
    import json
    with open(val_path, 'w') as f:
        json.dump({
            'linear_friendliness': val_linear,
            'causal_assumptions': val_causal
        }, f, indent=2)
    
    print(f"Saved causal features to {save_path}")
    print(f"Linear R2: {val_linear['linear_R2']:.3f}")
    print(f"Linearity ratio: {val_linear['linearity_ratio']:.3f}")
    print(f"Max standardized difference: {val_linear['max_std_diff']:.3f}")
    print(f"A-M residual correlation: {val_causal.get('residual_correlation_AM', np.nan):.3f}")
    
    return phi_df


if __name__ == "__main__":
    # Example usage
    print("Causal Linear-Friendly Autoencoder ready for import")
    print("Use train_causal_linear_ae() to train the model")
    print("Use validate_linear_friendliness() to check feature quality")
    print("Use export_causal_features() to save for R analysis")