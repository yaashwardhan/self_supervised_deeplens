# =================== Finetuning Rotation =========================== 

finetuning_model_rotation = keras.Sequential(
    [
        pretraining_model.encoder,
        layers.Dense(3, activation='softmax'),
    ],
    name="finetuning_model_rotation",
)
finetuning_model_rotation.compile(
    optimizer=keras.optimizers.Adam(learning_rate= 1e-4),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['acc', metrics.AUC(name='auc')])

finetuning_history_rotation = finetuning_model_rotation.fit(train_generator_one_hot, steps_per_epoch=steps_per_epoch, epochs=num_epochs, 
                                          validation_data=test_generator_one_hot, validation_steps = validation_steps)
print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(finetuning_history_rotation.history["val_acc"]) * 100
    )
)


# =================== Finetuning Gaussian =========================== 

finetuning_model_gaussian = keras.Sequential(
    [
        pretraining_model_gaussian.encoder,
        layers.Dense(3, activation='softmax'),
    ],
    name="finetuning_model",
)
finetuning_model_gaussian.compile(
    optimizer=keras.optimizers.Adam(learning_rate= 1e-4),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['acc', metrics.AUC(name='auc')])

finetuning_history_gaussian = finetuning_model_gaussian.fit(train_generator_one_hot, steps_per_epoch=steps_per_epoch, epochs=num_epochs, 
                                          validation_data=test_generator_one_hot, validation_steps = validation_steps)
print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(finetuning_history_gaussian.history["val_acc"]) * 100
    )
)