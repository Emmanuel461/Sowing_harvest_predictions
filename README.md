### **Group-Based Splitting**

The methodology employed in the pipeline focuses on how the separation of training and testing data, cross-validation, and hyperparameter tuning was implemented. Below, I provide a comprehensive explanation of the decisions taken and the reasoning behind them.

---

### **1. Group-Based Data Splitting for Training and Testing**

The implementation ensures that the data from the test group (e.g., a specific parcel) is entirely excluded from training. Rather than using `LeaveOneGroupOut` explicitly for splitting, a **manual mask-based approach** is employed to define the training and testing sets, ensuring group independence.

The following code demonstrates how the test group is explicitly excluded:

```python
# Define the group to be used as the test set
test_group = 3

# Create masks for training and testing
groups = location_flat_valid.astype(int)  # Group labels corresponding to parcels
train_mask = groups != test_group
test_mask = groups == test_group

# Split the data into training and testing sets
X_train = X[train_mask]
y_train = y[train_mask]
groups_train = groups[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]
groups_test = groups[test_mask]

print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Groups in training: {np.unique(groups_train)}")
print(f"Groups in testing: {np.unique(groups_test)}")
```

This separation guarantees that the model does not see data from the test parcel during training, ensuring an independent and reliable evaluation. The test group can be adjusted as needed to evaluate different parcels.

---

### **2. Cross-Validation with Group Constraints**

For hyperparameter tuning, **GroupKFold** is employed to ensure that data from the same parcel is not shared between training and validation folds. This respects the spatial correlation within each group (parcel) and prevents overfitting due to shared information.

The code below demonstrates the usage of `GroupKFold` for cross-validation:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=n_splits) # 4 

for fold, (train_idx_fold, val_idx_fold) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
    print(f"Fold {fold + 1}/4")
    print(f"  Training samples: {len(train_idx_fold)}")
    print(f"  Validation samples: {len(val_idx_fold)}")

    # Split the data into training and validation for the current fold
    X_fold_train, X_fold_val = X_train[train_idx_fold], X_train[val_idx_fold]
    y_fold_train, y_fold_val = y_train[train_idx_fold], y_train[val_idx_fold]

    # Continue with normalization, model training, and validation as shown in the pipeline
```

This approach ensures the validation data for each fold remains independent, mirroring the real-world scenario where the model predicts on unseen parcels.

---

### **3. Why Not Repeated K-Fold Cross-Validation?**

**Repeated K-Fold Cross-Validation** was avoided because it does not account for the group structure in the data. Since the primary goal is to prevent any leakage of information between training and validation sets from the same parcel, using `GroupKFold` is essential.

Repeated K-Fold would lead to data from the same parcel being split across folds, violating the independence assumption. By contrast, GroupKFold respects the group structure and ensures proper evaluation.

---

### **4. Hyperparameter Tuning and Average Performance Metrics**

For hyperparameter tuning, a grid search is performed over the specified hyperparameter space using cross-validation folds created by `GroupKFold`. The metrics from each fold are averaged to identify the best hyperparameter combination. Below is the relevant code:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=n_splits)

grid = list(ParameterGrid(param_grid))

# Variables para almacenar los mejores hiperparámetros y score
best_params = None
best_score = np.inf  # Porque queremos minimizar el RMSE

for params in grid:
    print(f"\nEvaluating hyperparameters: {params}")
    fold_metrics = []

    # Configure the current hyperparameters
    dropout_rate = params['dropout_rate']
    learning_rate = params['learning_rate']
    optimizer_name = params['optimizer']

    for fold, (train_idx_fold, val_idx_fold) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        print(f"  Fold {fold + 1}/{n_splits}")
        print(f"    Número de muestras en entrenamiento: {len(train_idx_fold)}")
        print(f"    Número de muestras en validación: {len(val_idx_fold)}")

        # Check that the validation set is not empty
        if len(val_idx_fold) == 0:
            print(f"    Fold {fold + 1} skipped due to lack of validation samples.")
            continue

        # Split the data into training and validation
        X_fold_train, X_fold_val = X_train[train_idx_fold], X_train[val_idx_fold]
        y_fold_train, y_fold_val = y_train[train_idx_fold], y_train[val_idx_fold]

        # Normalize the data based on X_fold_train
        num_bands = X_fold_train.shape[1]
        num_images = X_fold_train.shape[2]

        means = np.mean(X_fold_train, axis=(0, 2))
        stds = np.std(X_fold_train, axis=(0, 2))
        stds[stds == 0] = 1  # Avoid division by zero

        def normalize_data(X, means, stds):
            return (X - means[np.newaxis, :, np.newaxis]) / stds[np.newaxis, :, np.newaxis]

        X_fold_train_norm = normalize_data(X_fold_train, means, stds)
        X_fold_val_norm = normalize_data(X_fold_val, means, stds)

        # Create datasets and dataloaders
        train_dataset = PixelDataset(X_fold_train_norm, y_fold_train)
        val_dataset = PixelDataset(X_fold_val_norm, y_fold_val)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Definir el modelo con los hiperparámetros actuales
        model = CNN3DModelPixelwise(num_bands=num_bands, num_images=num_images, dropout_rate=dropout_rate).to(device)

        #  Define the loss and the optmizer 
        criterion = nn.L1Loss()
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'Adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

        # Entrenar el modelo en X_fold_train
        num_epochs = 60  # Ajusta según sea necesario
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluate the model on the training set to calculate R²
        model.eval()
        all_train_preds = []
        all_train_trues = []
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

                preds = outputs.cpu().numpy()
                trues = targets.cpu().numpy()

                all_train_preds.extend(preds)
                all_train_trues.extend(trues)

        # Evaluate the model on the validation set
        all_val_preds = []
        all_val_trues = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

                preds = outputs.cpu().numpy()
                trues = targets.cpu().numpy()

                all_val_preds.extend(preds)
                all_val_trues.extend(trues)

        # Calculate metrics for the validation set
        if len(all_val_trues) > 0:
            val_rmse = np.sqrt(mean_squared_error(all_val_trues, all_val_preds))
            val_mae = mean_absolute_error(all_val_trues, all_val_preds)
            val_bias = np.mean(np.array(all_val_preds) - np.array(all_val_trues))
            val_pbias = 100 * np.sum(np.array(all_val_preds) - np.array(all_val_trues)) / np.sum(np.array(all_val_trues))
        else:
            val_rmse = np.nan
            val_mae = np.nan
            val_bias = np.nan
            val_pbias = np.nan

        print(f"    RMSE en validación: {val_rmse:.4f}")
        print(f"    MAE en validación: {val_mae:.4f}")
        print(f"    Bias en validación: {val_bias:.4f}")
        print(f"    PBIAS en validación: {val_pbias:.4f}%")

        # Store the metrics for this fold
        fold_metric = {
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_bias': val_bias,
            'val_pbias': val_pbias
        }
        fold_metrics.append(fold_metric)

    # Calculate the average metrics across all folds
    mean_metrics = {
        'mean_val_rmse': np.nanmean([m['val_rmse'] for m in fold_metrics]),
        'mean_val_mae': np.nanmean([m['val_mae'] for m in fold_metrics]),
        'mean_val_bias': np.nanmean([m['val_bias'] for m in fold_metrics]),
        'mean_val_pbias': np.nanmean([m['val_pbias'] for m in fold_metrics]),
    }

    # Print the average metrics
    print(f"Resultados promedio para los hiperparámetros {params}:")
    print(f"  RMSE promedio en validación: {mean_metrics['mean_val_rmse']:.4f}")
    print(f"  MAE promedio en validación: {mean_metrics['mean_val_mae']:.4f}")
    print(f"  Bias promedio en validación: {mean_metrics['mean_val_bias']:.4f}")
    print(f"  PBIAS promedio en validación: {mean_metrics['mean_val_pbias']:.4f}%")

    # Update the best hyperparameters if needed
    if mean_metrics['mean_val_rmse'] < best_score:
        best_score = mean_metrics['mean_val_rmse']
        best_params = params

print(f"\nMejores hiperparámetros encontrados: {best_params} con RMSE promedio en validación de {best_score:.4f}")
```

The averaging of metrics ensures a robust selection of hyperparameters that generalize well across all parcels.

---

### **5. Final Evaluation**

Once the best hyperparameters are identified, the model is retrained on the full training set and evaluated on the held-out test set (e.g., parcel 3). Metrics such as RMSE, MAE, R², bias, and PBIAS are calculated to assess the model's performance.

The following code snippet demonstrates the evaluation:

```python
## ....
# Evaluate the model on the test set
model.eval()
all_test_predictions = []
all_test_true_values = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        preds = outputs.cpu().numpy()
        trues = targets.cpu().numpy()

        all_test_predictions.extend(preds)
        all_test_true_values.extend(trues)

# Calculate metrics on the test data
mse = mean_squared_error(all_test_true_values, all_test_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_test_true_values, all_test_predictions)
r2 = r2_score(all_test_true_values, all_test_predictions)
bias = np.mean(np.array(all_test_predictions) - np.array(all_test_true_values))
pbias = 100 * np.sum(np.array(all_test_predictions) - np.array(all_test_true_values)) / np.sum(np.array(all_test_true

print('\nMétricas en el Conjunto de Prueba:')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R²: {r2:.4f}')
print(f'Bias: {bias:.4f}')
print(f'PBIAS: {pbias:.4f}%')
# ....

```

---

### **Conclusion**

The methodology adopted ensures:
1. Independent evaluation of the model on unseen parcels using group-based splitting.
2. Reliable hyperparameter tuning with `GroupKFold` cross-validation.
3. Prevention of data leakage by respecting the group structure in all stages of the pipeline.

By employing these techniques, the model's performance is evaluated robustly and reflects its real-world application potential.


